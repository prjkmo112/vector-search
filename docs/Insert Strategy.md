# Insert Strategy

## [Insert Vector v1](../src/vector_search/insert_vector_single.py)
### Vector
- `sparse`  
    Product name
- `dense`  
    Product Image

## [Insert Vector v2](../src/vector_search/insert_vector_expand.py)
### Collections
- commerce_product
  - image_dense
    - 512 dim, Cosine, `CLIP ViT-B/32`
    - product image
  - text_property_dense
    - 1024 dim, Cosine, `BGE-M3`
    - title + category + brand + maker + model + keywords
    - `{title} {category} {brand} {maker} {model} {keywords}`
  - text_title_sparse
    - TF-IDF
    - title

- _commerce_product_docs: todo.._