"""Native fused sufficient-statistics and recursive least-squares updates."""

from libc.math cimport isfinite
from libc.stdlib cimport free, malloc


cdef inline void _symmetric_rank_one(
    double[:, ::1] matrix,
    const double[::1] vector,
    double scale,
    Py_ssize_t p,
) noexcept nogil:
    """Apply ``matrix += scale * vector vector.T`` and preserve exact symmetry."""

    cdef Py_ssize_t i, j
    cdef double value
    for i in range(p):
        for j in range(i + 1):
            # Averaging also removes asymmetry introduced by the initial dense
            # solve. Writing both triangles makes symmetry a state invariant.
            value = (
                0.5 * (matrix[i, j] + matrix[j, i])
                + scale * vector[i] * vector[j]
            )
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
    cdef double value

    for i in range(p):
        xty[i] += signed_weight * row[i] * target
        for j in range(i + 1):
            value = xtx[i, j] + signed_weight * row[i] * row[j]
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
    cdef double* work
    cdef double[::1] work_view
    cdef bint valid = True

    if xtx.shape[0] != p or xtx.shape[1] != p:
        raise ValueError("xtx shape must match the row width.")
    if inverse.shape[0] != p or inverse.shape[1] != p:
        raise ValueError("inverse shape must match the row width.")
    if xty.shape[0] != p or beta.shape[0] != p:
        raise ValueError("vector state shape must match the row width.")
    if targets.shape[0] != n or weights.shape[0] != n:
        raise ValueError("targets and weights must have one value per row.")

    work = <double*>malloc(p * sizeof(double))
    if work == NULL:
        raise MemoryError("Could not allocate native rank-one update workspace.")
    work_view = <double[:p]>work
    try:
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

                _matrix_vector(inverse, rows[row_index], work, p)
                projection = 0.0
                residual = targets[row_index]
                for i in range(p):
                    projection += rows[row_index, i] * work[i]
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
                    beta[i] += scale * work[i] * residual
                _symmetric_rank_one(inverse, work_view, -scale, p)
                inverse_updates += 1
    finally:
        free(work)

    return processed, valid, inverse_updates


def stats_add(
    double[:, ::1] xtx,
    double[::1] xty,
    const double[::1] row,
    double target,
    double weight,
):
    """Add one row to sufficient statistics without an initialized inverse."""

    cdef Py_ssize_t p = row.shape[0]
    if xtx.shape[0] != p or xtx.shape[1] != p or xty.shape[0] != p:
        raise ValueError("statistics shape must match the row width.")
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
    if old_row.shape[0] != p:
        raise ValueError("old and new rows must have the same width.")
    if xtx.shape[0] != p or xtx.shape[1] != p or xty.shape[0] != p:
        raise ValueError("statistics shape must match the row width.")
    with nogil:
        _stats_rank_one(xtx, xty, old_row, old_target, old_weight, -1.0, p)
        _stats_rank_one(xtx, xty, new_row, new_target, new_weight, 1.0, p)


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
    cdef double denominator, projection, residual, scale
    cdef double* work
    cdef double[::1] work_view
    cdef bint valid = True

    if old_row.shape[0] != p:
        raise ValueError("old and new rows must have the same width.")
    if xtx.shape[0] != p or xtx.shape[1] != p:
        raise ValueError("xtx shape must match the row width.")
    if inverse.shape[0] != p or inverse.shape[1] != p:
        raise ValueError("inverse shape must match the row width.")
    if xty.shape[0] != p or beta.shape[0] != p:
        raise ValueError("vector state shape must match the row width.")

    work = <double*>malloc(p * sizeof(double))
    if work == NULL:
        raise MemoryError("Could not allocate native rolling update workspace.")
    work_view = <double[:p]>work
    try:
        with nogil:
            _stats_rank_one(
                xtx, xty, old_row, old_target, old_weight, -1.0, p
            )
            _stats_rank_one(
                xtx, xty, new_row, new_target, new_weight, 1.0, p
            )

            if old_weight != 0.0:
                _matrix_vector(inverse, old_row, work, p)
                projection = 0.0
                residual = old_target
                for i in range(p):
                    projection += old_row[i] * work[i]
                    residual -= old_row[i] * beta[i]
                denominator = 1.0 - old_weight * projection
                if (
                    not isfinite(denominator)
                    or denominator <= min_denominator
                ):
                    valid = False
                else:
                    scale = old_weight / denominator
                    for i in range(p):
                        beta[i] -= scale * work[i] * residual
                    _symmetric_rank_one(inverse, work_view, scale, p)

            if valid and new_weight != 0.0:
                _matrix_vector(inverse, new_row, work, p)
                projection = 0.0
                residual = new_target
                for i in range(p):
                    projection += new_row[i] * work[i]
                    residual -= new_row[i] * beta[i]
                denominator = 1.0 + new_weight * projection
                if (
                    not isfinite(denominator)
                    or denominator <= min_denominator
                ):
                    valid = False
                else:
                    scale = new_weight / denominator
                    for i in range(p):
                        beta[i] += scale * work[i] * residual
                    _symmetric_rank_one(inverse, work_view, -scale, p)
    finally:
        free(work)

    return valid
