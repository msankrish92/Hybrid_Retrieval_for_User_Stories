from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
import requests
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from groq import Groq


class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


model = GroqModel(
    model="llama-3.1-8b-instant",
    api_key="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"
)

# Initialize metrics with Groq
answer_relevancy = AnswerRelevancyMetric(model=model, threshold=0.7)
# faithfulness = FaithfulnessMetric(model=model, threshold=0.7)


def test_user_story_validation():
    """Test the actual user story validation endpoint"""

    # Test input
    test_story = "As a user, I want to login to the system so that I can access my account."

    # Call your API (make sure server is running)
    try:
        response = requests.post(
            "http://localhost:3000/api/validate-story",
            json={"userStory": test_story},
            timeout=30
        )
        result = response.json()
        print("API Response:", result)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your server is running: node src/server/server.js")
        return

    # Extract context from related stories
    retrieval_context = []
    if "relatedStories" in result:
        for story in result["relatedStories"][:3]:
            retrieval_context.append(story.get("story", ""))

    # Create test case
    test_case = LLMTestCase(
        input=test_story,
        actual_output=str(result.get("validation", "")),
        expected_output="Should provide scores for Format, Clarity, Testability, Completeness, Consistency, and Grammar",
        retrieval_context=retrieval_context
    )

    # Evaluate
    # evaluate([test_case], [answer_relevancy, faithfulness])
    evaluate([test_case], [answer_relevancy])


if __name__ == "__main__":
    test_user_story_validation()

# from deepeval.metrics import FaithfulnessMetric
# faithfulness = FaithfulnessMetric(model=model, threshold=0.7)
