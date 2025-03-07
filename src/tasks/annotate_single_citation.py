import os
from src.paper import load_papers_from_json

# Load papers
papers = load_papers_from_json("./data/paper_obj_raw")
for paper in papers:
    for section_name, section in paper.sections.items():
        # Get citations in section
        citations = paper.sections[section_name]['citations']
        previous_citation = None
        next_citation = None
        for i, citation in enumerate(citations):
            single_citation = True
            if i > 0:
                previous_citation = citations[i-1]
            if i < len(citations) - 1:
                next_citation = citations[i+1]

            if i == 0:
                previous_citation = None

            if i == len(citations) -1:
                next_citation = None
            
            if previous_citation:
                if citation.start_pos - previous_citation.end_pos < 4:
                    single_citation = False
            
            if next_citation:
                if next_citation.start_pos - citation.end_pos < 4:
                    single_citation = False
            citation.single_citation = single_citation

def save_papers(papers, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for paper in papers:
        arxiv_id = paper.paper_id.split('/')[-1].split("v")[0]
        output_filename = f"{arxiv_id}.json"
        output_path = os.path.join(output_dir, output_filename)
        # Save paper to JSON
        paper.save_to_file(output_path)

save_papers(papers, "./data/paper_obj_anno")