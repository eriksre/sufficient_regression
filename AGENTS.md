Always activate `.venv` before running Python.

Whenever you run into a bug or error to do with this repos code, please add it to the bugs.md file in docs here: docs/bugs.md. You should add any fixed bugs to docs/fixed_bugs.md. 

When writing code, if there is an explicit reason you've chosen to use something / design something in some way. ie, any assumptions you've made, then please document those assumptions clearly in the code. Keep this reasoning succinct. 

If you need context about the goal/objective of this project, that is available in the README.md file. 

Feel free to delegate work to subagents / new threads. Make sure to clean up those subagents when they have finished their work. Always make sure to specify the extra high reasoning effort for new subagents and new threads. 

The code you write code should fail loudly. 

Fallbacks and fallback ways of doing things should be rare in this codebase. When we do something, we do it right, and we don't add a fallback that sometimes works, or that makes code harder to debug. When fallbacks are implemented, their existence must be clearly justified in the code as comments / docstrings. 

Do not support legacy code and legacy methods. 

Make sure the code you write works in live testing. 

Do not delete or modify anything marked as do not modify or delete

Make sure you commit and push any changes you've made to main when you're finished. 
