def select_diverse_examples(texts, k=5, model_name="all-MiniLM-L6-v2"):
    """
    Selects k diverse examples using KMeans clustering on sentence embeddings.

    Args:
        texts (List[str]): All text examples from a single class.
        k (int): Number of diverse examples to return.
        model_name (str): Sentence embedding model to use.

    Returns:
        List[str]: Selected diverse examples.
    """
    # Step 1: Load embedding model and embed texts
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Step 2: Run KMeans clustering
    if len(texts) < k:
        print(f"Fewer examples ({len(texts)}) than requested clusters ({k}).")
        return texts

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # Step 3: For each cluster, pick the closest example to its centroid
    selected_indices = []
    for cluster_id in range(k):
        # Get all indices belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = cluster_centers[cluster_id]

        # Find the example closest to the centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        selected_indices.append(closest_index)

    # Step 4: Return the selected diverse examples
    return [texts[i] for i in selected_indices]



def select_uncertain_examples(texts, labels, target_label, model_pipeline, k=5, candidate_labels=["Req", "Not_Req"]):
    """
    Selects k most uncertain examples for a given class using a classification model.

    Args:
        texts (List[str]): All examples.
        labels (List[str]): Corresponding class labels.
        target_label (str): The class to filter (e.g., "Req").
        model_pipeline: A Hugging Face zero-shot or classification pipeline.
        k (int): Number of examples to return.
        candidate_labels (List[str]): List of possible labels (usually 2 classes).

    Returns:
        List[str]: The k most uncertain examples of the given class.
    """
    # Filter for the desired class
    label_texts = [text for text, lbl in zip(texts, labels) if lbl == target_label]

    if len(label_texts) < k:
        print(f" Only {len(label_texts)} examples in class '{target_label}'.")
        return label_texts

    uncertainty_scores = []

    # Classify each example and compute margin between top-2 scores
    for text in label_texts:
        try:
            result = model_pipeline(text, candidate_labels, multi_label=False)
            scores = result["scores"]

            # Sort scores to get top-2
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0] - sorted_scores[1]  # Smaller margin → more uncertain
            else:
                margin = 1.0  # Assume confident if only 1 class
        except Exception as e:
            print(f"Failed to classify text '{text[:30]}...': {e}")
            margin = 1.0  # Treat as confident if failure

        uncertainty_scores.append((margin, text))

    # Sort by ascending margin (lowest = most uncertain)
    sorted_by_uncertainty = sorted(uncertainty_scores, key=lambda x: x[0])

    # Return the k most uncertain examples
    return [text for _, text in sorted_by_uncertainty[:k]]
