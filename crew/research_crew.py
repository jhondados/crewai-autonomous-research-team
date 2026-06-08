"""CrewAI autonomous research team."""
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-1.5-pro-002")
search = SerperDevTool()

research_director = Agent(role="Research Director", goal="Orchestrate the research team to produce institutional-quality reports",
    backstory="Former Goldman Sachs MD with 20 years in equity research. Expert at structuring research and managing analyst teams.",
    llm=llm, tools=[search], verbose=True, allow_delegation=True)

market_analyst = Agent(role="Senior Market Analyst", goal="Analyze market size, trends, growth drivers and forecasts",
    backstory="Ex-McKinsey consultant with deep expertise in market sizing and competitive analysis across tech, fintech and healthcare.",
    llm=llm, tools=[search, WebsiteSearchTool()], verbose=True)

def create_research_crew(company: str, sector: str) -> Crew:
    tasks = [
        Task(description=f"Research {company} in {sector}: market position, revenue, growth trajectory",
             agent=market_analyst, expected_output="Detailed market analysis with data"),
        Task(description=f"Compile all research into executive report for {company}",
             agent=research_director, expected_output="Complete 30-page research report"),
    ]
    return Crew(agents=[research_director, market_analyst], tasks=tasks,
                process=Process.hierarchical, manager_llm=llm, verbose=True)
