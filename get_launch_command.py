import json
import os
import re

# launch.json のパスを指定
launch_json_path = "/workspaces/Ms2z/.vscode/launch.json"

def clean_json(json_str):
    """JSON内のコメント（// や /* */）を削除し、余分なカンマを修正"""
    # コメントを削除
    json_str = re.sub(r"//.*", "", json_str)  # 行コメント削除
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)  # ブロックコメント削除

    # 余分なカンマを削除
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)  # },] の前のカンマを削除

    return json_str

def get_command(target_name):
    if not os.path.exists(launch_json_path):
        return f"Error: launch.json not found at {launch_json_path}"

    with open(launch_json_path, "r", encoding="utf-8") as f:
        json_content = f.read()
    
    # コメントを削除
    json_content = clean_json(json_content)

    try:
        config = json.loads(json_content)  # コメント削除後にJSONデコード
    except json.JSONDecodeError as e:
        return f"JSONDecodeError: {e}"

    for entry in config["configurations"]:
        if entry["name"] == target_name:
            program = entry["program"]
            args = " ".join(entry.get("args", []))  # 引数をスペース区切りに
            return f"python {program} {args}"

    return f"No configuration found for '{target_name}'"

if __name__ == "__main__":
    target_name = input("Enter configuration name: ").strip()
    print(get_command(target_name))
