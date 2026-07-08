def get_wp_prompts(words, prompt):
    return [
        f'Write a story in {words} words to the prompt "{prompt}."',
        f'You are an author, who is writing a story in response to the prompt "{prompt}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word story on the following prompt: "{prompt}." Could you please draft something for me?',
        f'Please help me write a short story in response to the prompt "{prompt}."',
        f'Write a {words}-word story in the style of a beginner writer in response to the prompt "{prompt}."',
        f'Write a story with very short sentences in {words} words to the prompt "{prompt}."',
        f'Write a story in {words} words to the prompt "{prompt}." Do not use any markdown formatting (no asterisks, headers, or bullet points) — write in plain prose only, as a human would.',
    ]


def get_reuter_prompts(words, headline):
    return [
        f'Write a news article in {words} words based on the headline "{headline}."',
        f'You are a news reporter, who is writing an article with the headline "{headline}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word news article based on the following headline: "{headline}." Could you please draft something for me?',
        f'Please help me write a New York Times article for the headline "{headline}."',
        f'Write a {words}-word news article in the style of a New York Times article based on the headline "{headline}."',
        f'Write a news article with very short sentences in {words} words based on the headline "{headline}."',
        f'Write a news article in {words} words based on the headline "{headline}." Do not use any markdown formatting (no asterisks, headers, or bullet points) — write in plain prose only, as a human would.',
    ]


def get_essay_prompts(words, prompt):
    return [
        f'Write an essay in {words} words to the prompt "{prompt}."',
        f'You are a student, who is writing an essay in response to the prompt "{prompt}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word essay based on the following prompt: "{prompt}." Could you please draft something for me?',
        f'Please help me write an essay in response to the prompt "{prompt}."',
        f"Write a {words}-word essay in the style of a high-school student  in response to the following prompt: {prompt}.",
        f'Write an essay with very short sentences in {words} words to the prompt "{prompt}."',
        f'Write an essay in {words} words to the prompt "{prompt}." Do not use any markdown formatting (no asterisks, headers, or bullet points) — write in plain prose only, as a human would.',
    ]