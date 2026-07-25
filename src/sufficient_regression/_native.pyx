"""Native fused sufficient-statistics and recursive least-squares updates."""

from libc.math cimport isfinite


# The Python estimators switch to BLAS at 256 parameters, so their native
# scalar kernels never need more than this bounded workspace. Fixed stack
# buffers avoid one heap allocation on every streamed row.
DEF NATIVE_STACK_MAX_PARAMS = 256


# Streaming rows are checked for finiteness here rather than with a NumPy
# ``isfinite`` scan in the Python push path. The kernel already loads every
# element of the row, so the check is effectively free, whereas the NumPy scan
# it replaces cost more per row than the update arithmetic itself. Validation
# always runs before any state is mutated, so a rejected row leaves the
# estimator exactly as it was.
cdef enum:
    _VALID_OK = 0
    _VALID_ROW = 1
    _VALID_TARGET = 2


cdef inline int _validate_streaming_row(
    const double[::1] row,
    double target,
    Py_ssize_t p,
) noexcept nogil:
    cdef Py_ssize_t i
    if not isfinite(target):
        return _VALID_TARGET
    for i in range(p):
        if not isfinite(row[i]):
            return _VALID_ROW
    return _VALID_OK


cdef _raise_streaming_validation(int status):
    """Raise the same errors the NumPy validation path raised."""

    if status == _VALID_ROW:
        raise ValueError("x must contain only finite values.")
    if status == _VALID_TARGET:
        raise ValueError("y must contain only finite values.")
    raise AssertionError("Unknown streaming validation status.")


cdef inline void _symmetric_rank_one(
    double[:, ::1] matrix,
    const double[::1] vector,
    double scale,
    Py_ssize_t p,
) noexcept nogil:
    """Apply ``matrix += scale * vector vector.T``, preserving exact symmetry.

    The matrix is symmetric on entry as a state invariant: ``_ensure_inverse``
    projects the freshly built inverse to exact symmetry, and every update writes
    both triangles from a single computed value. Re-averaging the two triangles
    per element would therefore be a no-op costing an extra load, add and
    multiply in the innermost loop of the hot path.
    """

    cdef Py_ssize_t i, j
    cdef double row_scale, value
    for i in range(p):
        row_scale = scale * vector[i]
        for j in range(i + 1):
            value = matrix[i, j] + row_scale * vector[j]
            matrix[i, j] = value
            matrix[j, i] = value


cdef inline void _stats_rank_one(
    double[:, ::1] xtx,
    double[::1] xty,
    const double[::1] row,
    double target,
    double weight,
    double sign,
    Py_ssize_t p,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double signed_weight = sign * weight
    cdef double row_scale, value

    for i in range(p):
        # Hoisted out of the inner loop; multiplication associates left to right
        # so the products are bitwise identical to the unhoisted form.
        row_scale = signed_weight * row[i]
        xty[i] += row_scale * target
        for j in range(i + 1):
            value = xtx[i, j] + row_scale * row[j]
            xtx[i, j] = value
            xtx[j, i] = value


cdef inline void _stats_slide_fused(
    double[:, ::1] xtx,
    double[::1] xty,
    const double[::1] old_row,
    double old_target,
    double old_weight,
    const double[::1] new_row,
    double new_target,
    double new_weight,
    Py_ssize_t p,
) noexcept nogil:
    """Apply one remove/add pair while reading and writing each entry once."""

    cdef Py_ssize_t i, j
    cdef double value

    for i in range(p):
        value = xty[i] - old_weight * old_row[i] * old_target
        xty[i] = value + new_weight * new_row[i] * new_target
        for j in range(i + 1):
            value = xtx[i, j] - old_weight * old_row[i] * old_row[j]
            value += new_weight * new_row[i] * new_row[j]
            xtx[i, j] = value
            xtx[j, i] = value


cdef inline void _matrix_vector(
    const double[:, ::1] matrix,
    const double[::1] vector,
    double* output,
    Py_ssize_t p,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double value
    for i in range(p):
        value = 0.0
        for j in range(p):
            value += matrix[i, j] * vector[j]
        output[i] = value


cdef inline void _matrix_two_vectors(
    const double[:, ::1] matrix,
    const double[::1] first,
    const double[::1] second,
    double* first_output,
    double* second_output,
    Py_ssize_t p,
) noexcept nogil:
    """Multiply one matrix by both rolling rows in a single matrix read."""

    cdef Py_ssize_t i, j
    cdef double matrix_value, first_value, second_value
    for i in range(p):
        first_value = 0.0
        second_value = 0.0
        for j in range(p):
            matrix_value = matrix[i, j]
            first_value += matrix_value * first[j]
            second_value += matrix_value * second[j]
        first_output[i] = first_value
        second_output[i] = second_value


cdef inline void _symmetric_rank_two(
    double[:, ::1] matrix,
    const double[::1] old_direction,
    double old_scale,
    const double[::1] new_direction,
    double new_scale,
    Py_ssize_t p,
) noexcept nogil:
    """Fuse the rolling inverse downdate/update into one symmetric write.

    As in :func:`_symmetric_rank_one`, symmetry is a state invariant rather than
    something to re-establish per element: the freshly built inverse is projected
    to exact symmetry and every update writes both triangles from one computed
    value, so averaging the triangles here would be a no-op in the hot loop.
    """

    cdef Py_ssize_t i, j
    cdef double old_row_scale, new_row_scale, value
    for i in range(p):
        # Hoisted out of the inner loop; multiplication associates left to right
        # so the products are bitwise identical to the unhoisted form.
        old_row_scale = old_scale * old_direction[i]
        new_row_scale = new_scale * new_direction[i]
        for j in range(i + 1):
            value = matrix[i, j] + old_row_scale * old_direction[j]
            value -= new_row_scale * new_direction[j]
            matrix[i, j] = value
            matrix[j, i] = value


def append_update(
    double[:, ::1] xtx,
    double[::1] xty,
    double[:, ::1] inverse,
    double[::1] beta,
    const double[:, ::1] rows,
    const double[::1] targets,
    const double[::1] weights,
    double min_denominator,
):
    """Fuse batch statistics, inverse, and coefficient updates.

    Returns ``(processed_rows, inverse_valid, inverse_updates)``. If the
    denominator guard rejects an update, statistics for that row are already
    exact; callers can invalidate the inverse and finish any remaining
    statistics with the dense batch path.
    """

    cdef Py_ssize_t n = rows.shape[0]
    cdef Py_ssize_t p = rows.shape[1]
    cdef Py_ssize_t row_index, i
    cdef Py_ssize_t processed = 0
    cdef Py_ssize_t inverse_updates = 0
    cdef double denominator, residual, scale, projection
    cdef double work_storage[NATIVE_STACK_MAX_PARAMS]
    cdef double[::1] work_view
    cdef bint valid = True

    if p > NATIVE_STACK_MAX_PARAMS:
        raise ValueError("native stack kernel supports at most 256 parameters.")
    if xtx.shape[0] != p or xtx.shape[1] != p:
        raise ValueError("xtx shape must match the row width.")
    if inverse.shape[0] != p or inverse.shape[1] != p:
        raise ValueError("inverse shape must match the row width.")
    if xty.shape[0] != p or beta.shape[0] != p:
        raise ValueError("vector state shape must match the row width.")
    if targets.shape[0] != n or weights.shape[0] != n:
        raise ValueError("targets and weights must have one value per row.")

    work_view = <double[:p]>&work_storage[0]
    with nogil:
        for row_index in range(n):
            _stats_rank_one(
                xtx,
                xty,
                rows[row_index],
                targets[row_index],
                weights[row_index],
                1.0,
                p,
            )
            processed = row_index + 1
            if weights[row_index] == 0.0:
                continue

            _matrix_vector(
                inverse,
                rows[row_index],
                &work_storage[0],
                p,
            )
            projection = 0.0
            residual = targets[row_index]
            for i in range(p):
                projection += rows[row_index, i] * work_storage[i]
                residual -= rows[row_index, i] * beta[i]
            denominator = 1.0 + weights[row_index] * projection
            if (
                not isfinite(denominator)
                or denominator <= min_denominator
            ):
                valid = False
                break

            scale = weights[row_index] / denominator
            for i in range(p):
                beta[i] += scale * work_storage[i] * residual
            _symmetric_rank_one(inverse, work_view, -scale, p)
            inverse_updates += 1

    return processed, valid, inverse_updates


def validate_streaming_row(const double[::1] row, double target):
    """Check one streaming row for finiteness, raising as the kernel would.

    Wide systems take the NumPy/BLAS update path, which never enters a kernel
    where the row would otherwise be validated on the way through.
    """

    cdef int status = _validate_streaming_row(row, target, row.shape[0])
    if status != _VALID_OK:
        _raise_streaming_validation(status)


def append_update_one(
    double[:, ::1] xtx,
    double[::1] xty,
    double[:, ::1] inverse,
    double[::1] beta,
    const double[::1] row,
    double target,
    double weight,
    double min_denominator,
):
    """Fuse one appended row's statistics, inverse, and coefficient update.

    Returns ``(inverse_valid, inverse_updates)``. This exists alongside
    :func:`append_update` because the single-row streaming path would otherwise
    have to reshape the row and allocate length-one target and weight arrays on
    every push purely to satisfy a batch signature.
    """

    cdef Py_ssize_t p = row.shape[0]
    cdef Py_ssize_t i
    cdef double denominator, residual, scale, projection
    cdef double work_storage[NATIVE_STACK_MAX_PARAMS]
    cdef double[::1] work_view
    cdef bint valid = True
    cdef int status

    if p > NATIVE_STACK_MAX_PARAMS:
        raise ValueError("native stack kernel supports at most 256 parameters.")
    if xtx.shape[0] != p or xtx.shape[1] != p:
        raise ValueError("xtx shape must match the row width.")
    if inverse.shape[0] != p or inverse.shape[1] != p:
        raise ValueError("inverse shape must match the row width.")
    if xty.shape[0] != p or beta.shape[0] != p:
        raise ValueError("vector state shape must match the row width.")
    status = _validate_streaming_row(row, target, p)
    if status != _VALID_OK:
        _raise_streaming_validation(status)

    work_view = <double[:p]>&work_storage[0]
    with nogil:
        _stats_rank_one(xtx, xty, row, target, weight, 1.0, p)
        if weight != 0.0:
            _matrix_vector(inverse, row, &work_storage[0], p)
            projection = 0.0
            residual = target
            for i in range(p):
                projection += row[i] * work_storage[i]
                residual -= row[i] * beta[i]
            denominator = 1.0 + weight * projection
            if (
                not isfinite(denominator)
                or denominator <= min_denominator
            ):
                valid = False
            else:
                scale = weight / denominator
                for i in range(p):
                    beta[i] += scale * work_storage[i] * residual
                _symmetric_rank_one(inverse, work_view, -scale, p)

    return valid, 1 if (valid and weight != 0.0) else 0


def stats_add(
    double[:, ::1] xtx,
    double[::1] xty,
    const double[::1] row,
    double target,
    double weight,
):
    """Add one row to sufficient statistics without an initialized inverse."""

    cdef Py_ssize_t p = row.shape[0]
    cdef int status
    if xtx.shape[0] != p or xtx.shape[1] != p or xty.shape[0] != p:
        raise ValueError("statistics shape must match the row width.")
    status = _validate_streaming_row(row, target, p)
    if status != _VALID_OK:
        _raise_streaming_validation(status)
    with nogil:
        _stats_rank_one(xtx, xty, row, target, weight, 1.0, p)


def stats_slide(
    double[:, ::1] xtx,
    double[::1] xty,
    const double[::1] old_row,
    double old_target,
    double old_weight,
    const double[::1] new_row,
    double new_target,
    double new_weight,
):
    """Apply one rolling remove/add pair to sufficient statistics."""

    cdef Py_ssize_t p = new_row.shape[0]
    cdef int status
    if old_row.shape[0] != p:
        raise ValueError("old and new rows must have the same width.")
    if xtx.shape[0] != p or xtx.shape[1] != p or xty.shape[0] != p:
        raise ValueError("statistics shape must match the row width.")
    # The departing row was validated when it entered the window, so only the
    # arriving row is checked. Both statistics updates happen after the check.
    status = _validate_streaming_row(new_row, new_target, p)
    if status != _VALID_OK:
        _raise_streaming_validation(status)
    with nogil:
        _stats_slide_fused(
            xtx,
            xty,
            old_row,
            old_target,
            old_weight,
            new_row,
            new_target,
            new_weight,
            p,
        )


def rolling_update(
    double[:, ::1] xtx,
    double[::1] xty,
    double[:, ::1] inverse,
    double[::1] beta,
    const double[::1] old_row,
    double old_target,
    double old_weight,
    const double[::1] new_row,
    double new_target,
    double new_weight,
    double min_denominator,
):
    """Fuse one rolling remove/add transition.

    The exact sufficient statistics are always updated. ``False`` means the
    downdate denominator was unsafe and the caller must invalidate the inverse.
    """

    cdef Py_ssize_t p = new_row.shape[0]
    cdef Py_ssize_t i
    cdef double denominator, projection, residual, coupling
    cdef double old_scale = 0.0
    cdef double new_scale = 0.0
    cdef double old_direction_storage[NATIVE_STACK_MAX_PARAMS]
    cdef double new_direction_storage[NATIVE_STACK_MAX_PARAMS]
    cdef double[::1] old_direction
    cdef double[::1] new_direction
    cdef bint valid = True
    cdef int status

    if p > NATIVE_STACK_MAX_PARAMS:
        raise ValueError("native stack kernel supports at most 256 parameters.")
    if old_row.shape[0] != p:
        raise ValueError("old and new rows must have the same width.")
    if xtx.shape[0] != p or xtx.shape[1] != p:
        raise ValueError("xtx shape must match the row width.")
    if inverse.shape[0] != p or inverse.shape[1] != p:
        raise ValueError("inverse shape must match the row width.")
    if xty.shape[0] != p or beta.shape[0] != p:
        raise ValueError("vector state shape must match the row width.")
    # The departing row was validated when it entered the window. Checking the
    # arriving row here, before any mutation, keeps a rejected push from leaving
    # partially updated statistics behind.
    status = _validate_streaming_row(new_row, new_target, p)
    if status != _VALID_OK:
        _raise_streaming_validation(status)

    old_direction = <double[:p]>&old_direction_storage[0]
    new_direction = <double[:p]>&new_direction_storage[0]
    with nogil:
        _stats_slide_fused(
            xtx,
            xty,
            old_row,
            old_target,
            old_weight,
            new_row,
            new_target,
            new_weight,
            p,
        )

        if old_weight != 0.0 or new_weight != 0.0:
            _matrix_two_vectors(
                inverse,
                old_row,
                new_row,
                &old_direction_storage[0],
                &new_direction_storage[0],
                p,
            )

        if old_weight != 0.0:
            projection = 0.0
            residual = old_target
            for i in range(p):
                projection += old_row[i] * old_direction_storage[i]
                residual -= old_row[i] * beta[i]
            denominator = 1.0 - old_weight * projection
            if (
                not isfinite(denominator)
                or denominator <= min_denominator
            ):
                valid = False
            else:
                old_scale = old_weight / denominator
                for i in range(p):
                    beta[i] -= (
                        old_scale
                        * old_direction_storage[i]
                        * residual
                    )

                # M_after_drop @ new_row is derived from the pre-slide
                # directions. This is algebraically identical to materializing
                # the downdated inverse and multiplying it by new_row.
                if new_weight != 0.0:
                    coupling = 0.0
                    for i in range(p):
                        coupling += old_direction_storage[i] * new_row[i]
                    coupling *= old_scale
                    for i in range(p):
                        new_direction_storage[i] += (
                            coupling * old_direction_storage[i]
                        )

        if valid and new_weight != 0.0:
            projection = 0.0
            residual = new_target
            for i in range(p):
                projection += new_row[i] * new_direction_storage[i]
                residual -= new_row[i] * beta[i]
            denominator = 1.0 + new_weight * projection
            if (
                not isfinite(denominator)
                or denominator <= min_denominator
            ):
                valid = False
            else:
                new_scale = new_weight / denominator
                for i in range(p):
                    beta[i] += (
                        new_scale
                        * new_direction_storage[i]
                        * residual
                    )

        if valid and (old_weight != 0.0 or new_weight != 0.0):
            _symmetric_rank_two(
                inverse,
                old_direction,
                old_scale,
                new_direction,
                new_scale,
                p,
            )

    return valid
