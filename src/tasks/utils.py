import os

def save_papers(papers, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for paper in papers:
        arxiv_id = paper.paper_id.split('/')[-1].split("v")[0]
        output_filename = f"{arxiv_id}.json"
        output_path = os.path.join(output_dir, output_filename)
        # Save paper to JSON
        paper.save_to_file(output_path)


def replace_substrings_in_batch(original_str, indexes, replacement):
    """
    Perform multiple non-overlapping substring replacements in `original_str`,
    where `replacements` is a list of (start, end, new_substring).
    
    Returns the modified string after all replacements.
    """
    
    # 1. Sort the replacements by start index
    indexes = sorted(indexes, key=lambda x: x[0])

    result_parts = []
    prev_end = 0

    # 2. Build the new string
    for (start, end) in indexes:
        # Append the segment of `original_str` that is before `start` but after the previous `end`
        result_parts.append(original_str[prev_end:start])

        # Append the replacement text
        result_parts.append(replacement)

        # Update where we left off
        prev_end = end

    # 3. Append anything left after the last replacement
    result_parts.append(original_str[prev_end:])

    # Join all parts into the final string
    return "".join(result_parts)


def convert_paper_id_to_arxiv_id(paper_id):
    return paper_id.split('/')[-1].split("v")[0]
