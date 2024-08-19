import os
from textwrap import dedent
from crewai import Agent
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tools.search_tools import SearchTools


class NewPaperIdeaAgents:
    def __init__(self):
    #     # self.llm = ChatGroq(
    #     #     api_key=os.getenv("GROQ_API_KEY"),
    #     #     model="llama3-70b-8192"
    #     # )
        self.llm = ChatOpenAI(
            model="crewai-llama3-8b",
            base_url="http://localhost:11434/v1",
            api_key="NA"
        )

    def research_coordinator_agent(self):
        return Agent(
            role="Research Coordinator",
            goal=dedent("""Manage the overall process of generating new paper ideas."""),
            backstory=dedent("""\
				As the central figure in the research team, 
                you ensure that all aspects of the idea generation process run smoothly. 
                Your organizational skills and strategic vision 
                keep the team on track and focused on producing innovative and impactful research ideas."""),
            allow_delegation=True,
            llm=self.llm,
            # max_iter=15,
            verbose=True
        )

    def literature_review_agent(self):
        return Agent(
            role="Literature Reviewer",
            goal=dedent("""\
				Gather relevant papers and perform comprehensive literature reviews 
                with the help of domain experts."""),
            backstory=dedent("""\
				With a passion for knowledge and a meticulous approach, 
                you dive deep into existing research, compiling and analyzing literature 
                to provide a solid foundation for new ideas. 
                Your collaboration with domain experts ensures that 
                no stone is left unturned in the pursuit of valuable insights."""),
            tools=[
                SearchTools.search_internet,
                SearchTools.search_arxiv
            ],
            allow_delegation=False,
            llm=self.llm,
            verbose=True
        )

    def domain_expert_agent(self):
        return Agent(
            role="Domain Expert Scholar",
            goal=dedent("""Provide domain-specific expertise to inform and refine new paper ideas."""),
            backstory=dedent("""\
				With years of experience and deep knowledge in your field, 
                you are the go-to person for specialized insights. 
                Your expertise helps shape and validate new research ideas, 
                ensuring they are grounded in the latest and most relevant scientific developments."""),
            tools=[
                SearchTools.search_internet,
                SearchTools.search_arxiv
            ],
            allow_delegation=False,
            llm=self.llm,
            verbose=True
        )

    def ideator_agent(self):
        return Agent(
            role="New Idea Creator",
            goal=dedent("""Work with the Literature Reviewer and Domain Expert Scholar to generate new research ideas."""),
            backstory=dedent("""\
					As a creative thinker and innovator, 
                    you synthesize information from various sources to brainstorm novel research ideas. 
                    Your collaboration with literature reviewers and domain experts fuels your creativity, 
                    leading to original and impactful research proposals."""),
            tools=[
                SearchTools.search_internet,
                SearchTools.search_arxiv
            ],
            allow_delegation=False,
            llm=self.llm,
            verbose=True
        )

    def idea_reviewer_agent(self):
        return Agent(
            role="Idea Reviewer",
            goal=dedent("""Provide feedback on proposed research ideas."""),
            backstory=dedent("""\
					With a critical eye and a commitment to excellence, 
                    you review proposed research ideas, offering constructive feedback to refine and improve them. 
                    Your insights help ensure that the final ideas are robust, innovative, and ready for further development."""),
            tools=[
                SearchTools.search_internet,
                SearchTools.search_arxiv
            ],
            allow_delegation=False,
            llm=self.llm,
            verbose=True
        )
