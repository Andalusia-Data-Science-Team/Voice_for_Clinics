import logging
from core.config import Config
from model.llm_service import LLMService

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate medical questions based on transcribed medical content."""
    @staticmethod
    def generate(translated_text: str, is_conversation: bool = False):
        """
        Generate questions that the doctor should ask the patient.
        
        Args:
            translated_text: The translated medical text
            is_conversation: Whether this is a doctor-patient conversation
            
        Returns:
            Tuple of (questions_list, reasoning)
        """
        try:
            api_key = Config.OPENROUTER_API_KEY
            result = LLMService.generate_questions(
                translated_text=translated_text,
                api_key=api_key,
                is_conversation=is_conversation
            )
            
            if not result:
                logger.warning("Question generation returned empty result")
                return [], "Failed to generate questions"
            
            questions = result.get("questions", [])
            reasoning = result.get("reasoning", "")
            
            logger.info(f"Generated {len(questions)} questions ({sum(1 for q in questions if q.get('needs_asking'))} need asking)")
            
            return questions, reasoning
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return [], f"Error: {str(e)}"