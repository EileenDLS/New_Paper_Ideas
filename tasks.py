from crewai import Task
from textwrap import dedent

class NewPaperIdeaTasks:
	def gather_literature(self, agent, pdf_file, key_words):
		return Task(
			description=dedent(f"""\
				Analyze the given paper: {pdf_file}.
				Based on the content of the provided paper and the following keywords: {key_words},
      			search for and collect relevant papers and research articles from the following journals and databases:
            
				- Management Science
            	- Information Systems Research
           	 	- Management Information Systems Quarterly
            	- Unpublished Manuscript Database - SSRN
            	- Unpublished Manuscript Database - Arxiv
            	- Marketing Science
            	- Journal of Marketing Research
            	- Journal of Marketing
            	- Top consumer behavior/decision process journals in marketing such as JCR
            	- Journal on Computing
            	- Strategic Management Journal
            	- MSI Working Papers Series
            	- ICIS Conference Proceedings
            	- HICSS Conference Proceedings
            	- Journal of Finance
            	- Journal of Interactive Marketing
            	- Any other journals on the list here: https://jsom.utdallas.edu/the-utd-top-100-business-school-research-rankings/

				# Perform comprehensive literature reviews with the help of domain experts 
				to build a strong foundation of existing knowledge.
			"""),
			agent=agent,
			async_execution=True,
            expected_output="""A list of related paper titles, URLs, and a brief summary. Ensure there are at least 5 well-formatted articles.
                Example Output: 
                [
                    {  'title': 'Deep Graph Contrastive Representation Learning', 
                    'url': 'https://example.com/paper1', 
                    'summary': 'Propose a novel framework for unsupervised graph representation learning by leveraging a contrastive objective at the node level...'
                    }, 
                    {{...}}
                ]
            """
		)

	def literature_analysis(self, agent, context):
		return Task(
			description=dedent("""\
				Analyze the collected literature to identify key themes, trends, gaps, and opportunities in the existing research. 
				Summarize the findings in a detailed report.
				"""),
            agent=agent,
            async_execution=True,
            context=context,
			expected_output=dedent("""\
				A detailed analysis for each news story, including a title, abstract, methods to achieve and a "what needs to be improved" section. 
				There should be at least 30 articles, each following the proper format.
						  
                Example Output: 
                '## Deep Graph Contrastive Representation Learning\n\n
                **The abstract:**\n\n
                ** In this paper, we propose a novel framework for unsupervised graph representation learning by leveraging a contrastive objective at the node level. ...\n\n
                **The methods to achieve:**\n\n
                - Model learns representations by generating graph views using two proposed schemes \n\n
                - Removing edges and masking node features.... \n\n
                **what needs to be improved:** \n\n
				** the GRACE model may still give biased outputs, as the provided data itself may be strongly biased during the processes of the data collection, graph construction, etc..\n\n'
				""")
		)

	def domain_consultation(self, agent, literature_report, prompt_paper):
		return Task(
			description=dedent(f"""\
				Provide domain-specific insights and expertise based on the given analysis report and prompt:
            	Analysis report: {literature_report}
            	Prompt: {prompt_paper}
            
            	Using this information, perform the following:
            	- Validate the relevance and significance of identified themes and gaps in the literature.
            	- Suggest potential research directions.
            	- Ensure that the analysis aligns with the current state and needs of the field.
			"""),
			agent=agent,
			async_execution=True,
			expected_output="""Insights and recommendations from Domain Expert Scholar Agents.
            Example Output:
            {
                'insight_1': 'The identified theme of digital transformation is highly relevant, considering recent trends in technology adoption...',
                'insight_2': 'A significant gap exists in the literature regarding the impact of AI on consumer behavior...',
                ...
            }
        """
		)

	def generate_idea(self, agent, literature_report, prompt_paper, expert_insights, callback_function):
		return Task(description=dedent(f"""\
			Collaborate with the Literature Reviewer and Domain Expert Scholar to brainstorm and generate novel research ideas:
            Analysis report: {literature_report}
            Insights from Domain Expert Scholar Agents: {expert_insights}
            Prompt: {prompt_paper}
            
            Using this information, perform the following:
            - Propose innovative and impactful research topics.
            - Ensure the ideas align with the prompt and current research trends.
			"""),
			agent=agent,
			expected_output="""Multi-agent Desiderata output:
            Several ranked paper ideas including:

            1. **Title**: [Title of the Research Idea]
            2. **Abstract**: [Brief summary of the research idea]
            3. **Data and Methods**: [Description of the data sources and methods to achieve the research objectives]
            4. **Reasoning Based on Existing Papers**: [Explanation of how this idea fills a gap in the literature, including several existing paper citations and a paragraph of reasoning why this idea is novel and valuable]

            Example Output:
            [
                { 
                    'title': 'The Impact of AI on Consumer Behavior',
                    'abstract': 'This study will explore how artificial intelligence influences consumer purchasing decisions in digital marketplaces.',
                    'data_and_methods': 'Data will be collected from online consumer transactions and analyzed using machine learning algorithms to identify patterns and impacts.',
                    'reasoning': 'Existing studies like Smith et al. (2022) and Johnson (2023) have explored AI applications, but there is a gap in understanding its specific effects on consumer behavior. This research will address this gap by providing new insights into AI-driven consumer trends, which are not sufficiently covered in the current literature.'
                },
                { 
                    'title': 'Sustainable Business Practices in the Digital Age',
                    'abstract': 'This research will investigate how businesses can leverage digital technologies to implement and enhance sustainable practices.',
                    'data_and_methods': 'Case studies of businesses using digital tools for sustainability, along with quantitative analysis of their environmental impact.',
                    'reasoning': 'While previous works such as Lee (2021) and Brown (2022) discuss sustainability, they lack a focus on the role of digital technology in this context. This study will fill that gap by examining how digital solutions can drive sustainable practices, providing valuable insights for both researchers and practitioners.'
                },
                ...
            ]
        """
		)

	def review_idea(self, agent, proposed_ideas):
		return Task(description=dedent(f"""\
			Review the proposed research ideas:
            Proposed research ideas: {proposed_ideas}
            
            Provide constructive feedback and suggestions for improvement, focusing on:
            - Originality: Assess how innovative and unique the ideas are.
            - Feasibility: Evaluate the practicality of the proposed methods and data sources.
            - Potential Impact: Consider the significance and contribution of each idea to the field.
            - Ensure that the ideas are well-founded and have the potential for a substantial impact.
    		"""),
			agent=agent,
			expected_output="""Feedback provision.
            Example Output:
            [
                { 
                    'idea': 'The Impact of AI on Consumer Behavior',
                    'feedback': 'This idea is highly original and timely, given the current advancements in AI. However, the feasibility could be improved by specifying the types of machine learning algorithms to be used. The impact is significant as it addresses a gap in consumer behavior research related to AI. Consider focusing on specific consumer segments for more targeted insights.'
                },
                { 
                    'idea': 'Sustainable Business Practices in the Digital Age',
                    'feedback': 'The concept is relevant and aligns well with current sustainability trends. The feasibility of using case studies is good, but ensure that you have access to comprehensive data from multiple businesses. The potential impact is considerable, as it integrates digital technology with sustainability, which is an emerging area. Make sure to highlight specific technologies and their applications.'
                },
                ...
            ]
        """
		)

	# def finalize_idea(self, agent, feedback_provision, proposed_ideas):
	# 	return Task(description=dedent(f"""\
	# 		Oversee the finalization of the research ideas:
    #         Feedback from Reviewers: {feedback_provision}
    #         Proposed research ideas: {proposed_ideas}
            
    #         Incorporate the feedback from Idea Reviewer to refine and enhance the research ideas. Ensure that:
    #         - The ideas are clearly articulated and well-defined.
    #         - All aspects of the research ideas are aligned with the initial prompt and objectives.
    #         - The research ideas are ready for further development and proposal writing.
    #     	"""),
	# 		agent=agent,
	# 		expected_output="""Multi-agent Desiderata output:
    #         Several ranked paper ideas including:

    #         1. **Title**: [Title of the Finalized Research Idea]
    #         2. **Abstract**: [Refined summary of the research idea]
    #         3. **Data and Methods**: [Detailed description of the data sources and methods to achieve the research objectives]
    #         4. **Reasoning Based on Existing Papers**: [Explanation of how this idea fills a gap in the literature, including several existing paper citations and a paragraph of reasoning why this idea is novel and valuable]

    #         Example Output:
    #         [
    #             { 
    #                 'title': 'The Impact of AI on Consumer Behavior',
    #                 'abstract': 'This study will explore how artificial intelligence influences consumer purchasing decisions in digital marketplaces, utilizing advanced machine learning techniques to uncover patterns and insights.',
    #                 'data_and_methods': 'Data will be sourced from online consumer transactions and analyzed using both supervised and unsupervised machine learning algorithms to identify trends and impacts.',
    #                 'reasoning': 'Existing studies like Smith et al. (2022) and Johnson (2023) address AI applications but lack focus on specific impacts on consumer behavior. This research will fill that gap by providing targeted insights into AI-driven consumer trends, offering new perspectives and practical implications for marketers.'
    #             },
    #             { 
    #                 'title': 'Sustainable Business Practices in the Digital Age',
    #                 'abstract': 'This research will investigate how businesses can leverage digital technologies to implement and enhance sustainable practices, focusing on technology applications and their effects on environmental impact.',
    #                 'data_and_methods': 'Case studies will be conducted on businesses utilizing digital tools for sustainability, with quantitative analysis of their environmental metrics and impact.',
    #                 'reasoning': 'While Lee (2021) and Brown (2022) discuss sustainability, they do not focus on digital technology's role. This study will address that gap by integrating digital solutions with sustainability, providing new insights and actionable recommendations for businesses aiming to enhance their environmental performance.'
    #             },
    #             ...
    #         ]
    #     """
	# 	)