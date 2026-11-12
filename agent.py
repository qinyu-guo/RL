import os
import sys
import subprocess
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
LLM = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# 任务集：每个任务有「描述」和「测试（assert 断言）」
TASKS = [
    {
        "task": "写一个函数 get_middle，返回列表中间的元素；列表长度为偶数时返回中间两个数的平均值；空列表返回 None。",
        "test": "assert get_middle([1,2,3,4,5]) == 3 and get_middle([1,2,3,4]) == 2.5 and get_middle([]) is None",
    },
    {
        "task": "写一个函数 fizzbuzz，输入整数 n：3 的倍数返回 'Fizz'，5 的倍数返回 'Buzz'，同时是 3 和 5 的倍数返回'FizzBuzz'，否则返回字符串形式的 n。",
        "test": "assert fizzbuzz(15) == 'FizzBuzz' and fizzbuzz(9) == 'Fizz' and fizzbuzz(10) == 'Buzz' and fizzbuzz(7) == '7'",
    },
    {
        "task": "写一个函数 is_valid_brackets，判断字符串是否是合法括号序列（只含 ()[]{}，必须正确配对）。",
        "test": (
            "assert is_valid_brackets('()'), 'case1: () 应通过'\n"
            "assert is_valid_brackets('{[]}'), 'case2: {[]} 应通过'\n"
            "assert not is_valid_brackets('([)]'), 'case3: ([)] 不应通过'\n"
            "assert not is_valid_brackets('(]'), 'case4: (] 不应通过'"
        ),
    },
    {
        "task": "写一个函数 roman_to_int，把罗马数字转成整数（注意 IV=4、IX=9 这类减法规则）。",
        "test": "assert roman_to_int('III') == 3 and roman_to_int('IV') == 4 and roman_to_int('MCMXCIV') == 1994",
    },
]

def generate_code(messages):
    resp = LLM.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.2,
    )
    return resp.choices[0].message.content

def extract_code(text):
    # LLM 常输出 ```python ... ``` 包裹的代码块，这里把代码抠出来
    if "```" in text:
        for part in text.split("```")[1::2]:
            code = part.strip()
            if code.lower().startswith("python"):
                code = code[len("python"):].strip()
            if code:
                return code
    return text.strip()

def run_code(code, test):
    full = code + "\n\n" + test
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        ok = r.returncode == 0
        feedback = (r.stderr or r.stdout).strip()
        return ok, feedback
    finally:
        os.unlink(path)

def solve(task, max_iter=5):
    messages = [{
        "role": "user",
        "content": f"请用 Python 完成这个任务，只输出代码：\n{task['task']}",
    }]
    for i in range(max_iter):
        text = generate_code(messages)
        code = extract_code(text)
        ok, feedback = run_code(code, task["test"])
        if ok:
            return True, code, i + 1
        # 关键一步：把报错喂回去，让它自己修
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"运行报错如下，请修复：\n{feedback}"})
    return False, None, max_iter

if __name__ == "__main__":
    passed = 0
    for t in TASKS:
        ok, code, iters = solve(t)
        print(("✅" if ok else "❌"), t["task"], f"— {iters} 轮")
        passed += ok
    print(f"\n通过率：{passed}/{len(TASKS)}")