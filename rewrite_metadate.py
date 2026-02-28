import json

input_path = "extracted/metadata.jsonl"
output_path = "extracted/metadata.jsonl"  # overwrite in place

rows = []
with open(input_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "conditioning_image" in record:
            record["conditioning_file_name"] = record.pop("conditioning_image")
        rows.append(record)

with open(output_path, "w") as f:
    for record in rows:
        f.write(json.dumps(record) + "\n")

print(f"Done. Renamed 'conditioning_image' → 'conditioning_file_name' in {len(rows)} rows.")
