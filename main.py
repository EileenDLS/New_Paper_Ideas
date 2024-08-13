from agents import NewPaperIdeaAgents
from tasks import NewPaperIdeaTasks
from crewai import Crew, Process
from textwrap import dedent
from file_io import save_markdown
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

class NewIdeaCrew:
    def __init__(self, pdf_file, key_words, prompt_paper):
        self.pdf_file = pdf_file
        self.key_words = key_words
        self.prompt_paper = prompt_paper

    def run(self):
        tasks = NewPaperIdeaTasks()
        agents = NewPaperIdeaAgents()

        # Create Agents
        research_coordinator_agent = agents.research_coordinator_agent()
        literature_review_agent = agents.literature_review_agent()
        domain_expert_agent = agents.domain_expert_agent()
        ideator_agent = agents.ideator_agent()
        # idea_reviewer_agent = agents.idea_reviewer_agent()

        # Create Tasks
        collect_literature_task = tasks.gather_literature(
            literature_review_agent, self.pdf_file, self.key_words)
        analyze_literature_task = tasks.literature_analysis(
            literature_review_agent, [collect_literature_task])
        domain_consultation_task = tasks.domain_consultation(
            domain_expert_agent, [analyze_literature_task], self.prompt_paper)
        generate_idea_task = tasks.generate_idea(
            ideator_agent, [analyze_literature_task], self.prompt_paper, [domain_consultation_task], save_markdown)
        # review_idea_task = tasks.review_idea(
        #     idea_reviewer_agent, [generate_idea_task])
        # finalize_idea_task = tasks.finalize_idea(
        #     research_coordinator_agent, [review_idea_task], [generate_idea_task])
        
        # Create Crew responsible for Copy
        new_idea_crew = Crew(
            agents=[
                research_coordinator_agent,
                literature_review_agent,
                domain_expert_agent,
                ideator_agent
                # idea_reviewer_agent
            ],
            tasks=[
                collect_literature_task,
                analyze_literature_task,
                domain_consultation_task,
                generate_idea_task
                # review_idea_task
            ],
            verbose=True
            # process=Process.hierarchical,
            # Remove this when running locally. This helps prevent rate limiting with groq.
            # max_rpm=2
        )

        new_paper_ideas = new_idea_crew.kickoff()
        return new_paper_ideas

# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Welcome to New Paper Idea Crew")
    print('-------------------------------')
    pdf_file = input(
        dedent("""
      Input the website of papers: 
    """))
    key_words = input(
        dedent("""
      Please input key words: 
    """))
    prompt_paper = input(
        dedent("""
      What is kind of paper you want to propose?
    """))

    newidea_crew = NewIdeaCrew(pdf_file, key_words, prompt_paper)
    result = newidea_crew.run()
    print("\n\n########################")
    print("## Here are New Paper Ideas")
    print("########################\n")
    print(result)


# pdf_file = "iopscience.iop.org/article/10.1088/1742-6596/1712/1/012042/pdf"
# key_words = [
#     "digital transformation",
#     "data analytics",
#     "machine learning",
#     "e-commerce"]
# prompt_paper = "I am looking to ideate on new research papers that explore the intersection of digital transformation and consumer behavior, with a focus on how data analytics and machine learning techniques are being applied in the context of e-commerce. The new papers should provide novel insights and methodologies that can advance the current understanding in these areas."


# import ipywidgets as widgets
# from IPython.display import display

# upload = widgets.FileUpload(accept='.pdf', multiple=False)
# display(upload)

# # Handling the file after upload
# def handle_upload(change):
#     for filename, file_info in upload.value.items():
#         with open(filename, 'wb') as f:
#             f.write(file_info['content'])
#         print(f"File {filename} uploaded successfully.")

# upload.observe(handle_upload, names='value')
