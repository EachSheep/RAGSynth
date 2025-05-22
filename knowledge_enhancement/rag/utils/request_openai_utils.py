import backoff  # for exponential backoff
import openai
import asyncio
import numpy as np
from typing import Any
from retrying import retry

# @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

# @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
def completions_with_prob_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

async def dispatch_openai_chat_requests(
    client,
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str],
    timeout: int
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words,
            timeout = timeout
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, model_name, stop_words=None, max_new_tokens=None) -> None:
        self.model_name = model_name
        self.stop_words = stop_words
        self.max_new_tokens = max_new_tokens

    # @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, client, input_string, temperature = 0.0, timeout = 600, max_tokens=None):
        try:
            response = completions_with_backoff(
                    client,
                    model = self.model_name,
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
                            {"role": "user", "content": input_string}
                        ],
                    temperature = temperature,
                    top_p = 0.999,
                    stop = self.stop_words,
                    timeout = timeout,
                    max_tokens=max_tokens
            )
            generated_text = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            return generated_text, finish_reason
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            raise

    def generate(self, client, input_string, temperature = 0.0, timeout = 600, max_tokens=None):
        return self.chat_generate(client, input_string, temperature, timeout=timeout, max_tokens=max_tokens)
    
    # @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate_with_prob(self, client, input_string, top_logprobs, temperature = 0.0, timeout = 600):
        response = completions_with_prob_with_backoff(
                client,
                model = self.model_name,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
                        {"role": "user", "content": input_string}
                    ],
                temperature = temperature,
                top_p = 0.999,
                logprobs = True,
                top_logprobs = top_logprobs,
                stop = self.stop_words,
                timeout = timeout
        )
        
        generated_text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        
        logprob_objects = response.choices[0].logprobs.content
        logprobs = [np.exp(logprob_object.logprob) for logprob_object in logprob_objects]
        tokens = [logprob_object.token for logprob_object in logprob_objects]
        
        return generated_text, finish_reason, logprobs, tokens

    def generate_with_prob(self, client, input_string, top_logprobs = 1, temperature = 0.0, timeout = 600):
        return self.chat_generate_with_prob(client, input_string, top_logprobs, temperature, timeout=timeout)
    
    def batch_chat_generate(self, client, messages_list, temperature = 0.0, timeout = 600):
        open_ai_messages_list = []
        system_prompt = "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    client, open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words, timeout=timeout
            )
        )
        finish_reason = [x.choices[0].finish_reason.strip() for x in predictions]
        return [x.choices[0].message.content.strip() for x in predictions]

    def batch_generate(self, client, messages_list, temperature = 0.0, timeout = 600):
        return self.batch_chat_generate(client, messages_list, temperature, timeout=timeout)

    def get_logprobs(self, client, input_string, temperature=0.0, timeout=600):
        response = completions_with_backoff(
            client,
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
                {"role": "user", "content": input_string}
            ],
            temperature=temperature,
            top_p=1,
            stop=self.stop_words,
            logprobs=True,
            top_logprobs = 1,   # Get the log probabilities of the top 5 most likely tokens for each token.
            timeout=timeout
        )
        logprobs = response.choices[0].logprobs
        return logprobs

    def logprobs_generate(self, client, input_string, temperature=0.0, timeout=600):
        return self.get_logprobs(client, input_string, temperature, timeout=timeout)