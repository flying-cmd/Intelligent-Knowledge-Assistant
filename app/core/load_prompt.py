from pathlib import Path
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger  # Optional, but nicer with logging

def load_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt file and render its variable placeholders.
    :param name: Prompt filename without the .prompt suffix, for example image_summary
    :param **kwargs: Variables to render, for example root_folder="test file"
        and image_content=("previous text", "following text")
    :return: The final rendered prompt string
    """
    # 1. Build the prompt path.
    prompt_path = PROJECT_ROOT / 'prompts' / f'{name}.prompt'

    # 2. Validate that the file exists.
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path.absolute()}")

    # 3. Read the raw prompt text.
    raw_prompt = prompt_path.read_text(encoding='utf-8')

    # 4. Render placeholders when variables are provided.
    if kwargs:
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"Prompt rendered successfully. Replaced variables: {list(kwargs.keys())}")
        return rendered_prompt
    return raw_prompt



if __name__ == '__main__':
    # Test placeholder rendering with the same call style used by application code.
    root_folder = "hl3070-user-manual"  # File name to replace
    image_content = ("This is the text before the image", "This is the text after the image")
    # Pass every variable required by the .prompt placeholders.
    final_prompt = load_prompt(
        name='image_summary',
        root_folder=root_folder,  # Maps to {root_folder}
        image_content=image_content  # Maps to {image_content[0]} and {image_content[1]}
    )
    print("Rendered final prompt:")
    print(final_prompt)
