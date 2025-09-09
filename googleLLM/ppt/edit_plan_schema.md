
# Edit Plan JSON Schema (for the LLM)

The assistant must output ONLY JSON with this top-level shape:
```json
{
  "edits": [ { /* operations */ } ]
}
```

Supported operations:

1) **replace_text**
```json
{"op":"replace_text","slide_index":0,"find":"old","replace":"new","shape_idx":null,"case_sensitive":false,"first_only":false}
```

2) **add_bullets**
```json
{"op":"add_bullets","slide_index":1,"shape_idx":2,"bullets":["a","b"],"mode":"append","level":0}
```

3) **edit_table_cell**
```json
{"op":"edit_table_cell","slide_index":2,"table_idx":1,"row":0,"col":1,"new_text":"value"}
```

4) **delete_slide**
```json
{"op":"delete_slide","slide_index":4}
```

5) **add_image**
```json
{"op":"add_image","slide_index":3,"image_path":"path/to/local.png","left_in":1.0,"top_in":1.5,"width_in":4.0}
```

6) **add_new_slide**
```json
{"op":"add_new_slide","layout":"title_and_content","title":"Summary","bullets":["point 1","point 2"]}
```

Notes
- `slide_index` and shape/table indices come from the extractor output.
- Operations are applied in the given order.
- For images, you must supply a valid **local** path; remote URLs are not handled in this script.
