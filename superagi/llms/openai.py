import re
import openai
from openai import APIError, InvalidRequestError
from openai.error import RateLimitError, AuthenticationError, Timeout, TryAgain
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm

MAX_RETRY_ATTEMPTS = 6
MIN_WAIT = 45 # Seconds
MAX_WAIT = 600 # Seconds

def custom_retry_error_callback(retry_state):
    logger.info("OpenAi Exception:", retry_state.outcome.exception())
    return {"error": "ERROR_OPENAI", "message": "Open ai exception: "+str(retry_state.outcome.exception())}


class OpenAi(BaseLlm):
    def __init__(self, api_key, model="gpt-4", temperature=0.6, max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT"), top_p=1,
                 frequency_penalty=0,
                 presence_penalty=0, number_of_results=1):
        """
        Args:
            api_key (str): The OpenAI API key.
            model (str): The model.
            temperature (float): The temperature.
            max_tokens (int): The maximum number of tokens.
            top_p (float): The top p.
            frequency_penalty (float): The frequency penalty.
            presence_penalty (float): The presence penalty.
            number_of_results (int): The number of results.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.number_of_results = number_of_results
        self.api_key = api_key
        openai.api_key = api_key
        openai.api_base = get_config("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.context_error_retry = 0

    def get_source(self):
        return "openai"

    def get_api_key(self):
        """
        Returns:
            str: The API key.
        """
        return self.api_key

    def get_model(self):
        """
        Returns:
            str: The model.
        """
        return self.model

    @retry(
        retry=(
            retry_if_exception_type(RateLimitError) |
            retry_if_exception_type(Timeout) |
            retry_if_exception_type(TryAgain)
        ),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS), # Maximum number of retry attempts
        wait=wait_random_exponential(min=MIN_WAIT, max=MAX_WAIT),
        before_sleep=lambda retry_state: logger.info(f"{retry_state.outcome.exception()} (attempt {retry_state.attempt_number})"),
        retry_error_callback=custom_retry_error_callback
    )
    def chat_completion(self, messages, max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT")):
        """
        Call the OpenAI chat completion API.

        Args:
            messages (list): The messages.
            max_tokens (int): The maximum number of tokens.

        Returns:
            dict: The response.
        """
        try:
            # openai.api_key = get_config("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                n=self.number_of_results,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            content = response.choices[0].message["content"]
            self.context_error_retry = 0
            return {"response": response, "content": content}
        except RateLimitError as api_error:
            logger.info("OpenAi RateLimitError:", api_error)
            raise RateLimitError(str(api_error))
        except Timeout as timeout_error:
            logger.info("OpenAi Timeout:", timeout_error)
            raise Timeout(str(timeout_error))
        except TryAgain as try_again_error:
            logger.info("OpenAi TryAgain:", try_again_error)
            raise TryAgain(str(try_again_error))
        except AuthenticationError as auth_error:
            logger.info("OpenAi AuthenticationError:", auth_error)
            return {"error": "ERROR_AUTHENTICATION", "message": "Authentication error please check the api keys: "+str(auth_error)}
        except InvalidRequestError as invalid_request_error:
            err_str = str(invalid_request_error)
            logger.info("OpenAi InvalidRequestError:", invalid_request_error)

            if 'Please reduce the length of the messages or completion' in err_str and self.context_error_retry < 2:
                # Openai invalid request error: This model's maximum context length is 8192 tokens.
                # However, you requested 8641 tokens (4609 in the messages, 4032 in the completion).
                # Please reduce the length of the messages or completion.
                logger.info("Reducing max_token size and trying again!")
                self.context_error_retry += 1
                nums_in_err = list(map(int, re.findall(r'\d+', err_str)))
                max_context, current_context = nums_in_err[0], nums_in_err[1]
                context_diff = current_context - max_context
                return self.chat_completion(messages, max_tokens-context_diff-5)

            return {"error": "ERROR_INVALID_REQUEST", "message": "Openai invalid request error: " + err_str}
        except Exception as exception:
            logger.info("OpenAi Exception:", exception)
            return {"error": "ERROR_OPENAI", "message": "Open ai exception: "+str(exception)}

    def verify_access_key(self):
        """
        Verify the access key is valid.

        Returns:
            bool: True if the access key is valid, False otherwise.
        """
        try:
            models = openai.Model.list()
            return True
        except Exception as exception:
            logger.info("OpenAi Exception:", exception)
            return False

    def get_models(self):
        """
        Get the models.

        Returns:
            list: The models.
        """
        try:
            models = openai.Model.list()
            models = [model["id"] for model in models["data"]]
            models_supported = ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4-32k']
            models = [model for model in models if model in models_supported]
            return models
        except Exception as exception:
            logger.info("OpenAi Exception:", exception)
            return []
