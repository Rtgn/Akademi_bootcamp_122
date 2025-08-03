import json
import re

def fix_jsonl_format(input_file, output_file):
    """Fix JSONL file format by ensuring each JSON object is on a single line."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove any trailing commas and clean up the content
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    
    # Split the content into individual JSON objects
    # Look for complete JSON objects that start with { and end with }
    json_objects = []
    brace_count = 0
    current_object = ""
    in_string = False
    escape_next = False
    
    for char in content:
        if escape_next:
            current_object += char
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            current_object += char
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            
        if not in_string:
            if char == '{':
                if brace_count == 0:
                    current_object = char
                else:
                    current_object += char
                brace_count += 1
            elif char == '}':
                current_object += char
                brace_count -= 1
                if brace_count == 0:
                    # Complete JSON object found
                    json_objects.append(current_object.strip())
                    current_object = ""
            else:
                if brace_count > 0:
                    current_object += char
        else:
            if brace_count > 0:
                current_object += char
    
    # Process each JSON object
    formatted_lines = []
    for i, json_str in enumerate(json_objects):
        if not json_str.strip():
            continue
            
        try:
            # Clean up whitespace and newlines
            cleaned_json = re.sub(r'\s+', ' ', json_str.strip())
            
            # Parse the JSON to validate it
            data = json.loads(cleaned_json)
            
            # Format as single line with consistent spacing
            formatted_json = json.dumps(data, ensure_ascii=False, separators=(',', ': '))
            formatted_lines.append(formatted_json)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON object {i+1}: {e}")
            print(f"Content: {json_str[:200]}...")
            continue
    
    # Write the formatted content
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in formatted_lines:
            f.write(line + '\n')
    
    print(f"Successfully formatted {len(formatted_lines)} JSON objects")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    input_file = "postpartum_chatbot_100_examples.jsonl"
    output_file = "postpartum_chatbot_100_examples_fixed.jsonl"
    
    fix_jsonl_format(input_file, output_file) 