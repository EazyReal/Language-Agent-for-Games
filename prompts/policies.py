naive_prompt = """
Write a new Agent class referencing the starter code to interact with the environment and maximize your reward. 
For the coding part:
- Enclose the code by ```python and ```
- Only write the new Agent class, write comments to explain your reasoning.
- `import` need to be under the Agent scope. you can only import from the standard library and numpy.

The response format should be:
[Code]
```python
class Agent:
    ... # your code
```
""".strip()


cot_prompt = """
Now, solve the task with the following steps:
First, think about the problem and write down your thoughts:
- Recap some basic game theory knowledge (Nash equilibrium, best response) and apply it to the problem.
Second, write a new Agent class referencing the starter code to interact with the environment and maximize your reward. 
For the coding part:
- Enclose the code by ```python and ```
- Only write the new Agent class, write comments to explain your reasoning.
- `import` needs to be under the Agent scope. You can only import from the standard library and numpy.

The response format should be:
[Thoughts] <your thoughts>
[Code]
```python
class Agent:
    ... # your code
```
""".strip()
