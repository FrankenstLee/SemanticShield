"""Prompt builders for different datasets."""

from typing import List


def build_prompt(dataset: str, items: List[dict]) -> str:
    if dataset == "Clothing":
        prompt_items = "\n\n".join(
            [
                f"{i + 1}. Title: {item.get('title', 'N/A')}\n"
                f"   Categories: {', '.join([cat.replace('Clothing, Shoes & Jewelry', '').strip() for cat in item.get('categories', []) if isinstance(cat, str)])}"
                for i, item in enumerate(items)
            ]
        )
        input_template = (
            "You are a careful and intelligent behavioral reviewer in a recommender system.\n"
            "Attackers may inject fake users with fabricated interaction histories to manipulate item rankings or degrade recommendation performance.\n"
            "Your task is to evaluate a user's interaction history and decide whether they are a **real, human user** or a **fake/malicious user**.\n"
            "Provide two things:\n"
            "1. A detailed explanation of your reasoning.\n"
            "2. A final judgment: 'Real' or 'Fake'.\n"
            "Guidelines:\n"
            "In the Clothing, Shoes & Jewelry domain, real users often show consistent behavior — such as a strong preference for one gender’s products, or logical patterns (e.g. dress + heels + bag).\n"
            "Users who interact mostly with one category, brand, or gender are typically genuine. If a user meets this criterion, you should consider them real.\n"
            "Cross-gender activity is not inherently fake, but it becomes suspicious if the user interacts with both male and female products in a scattered, inconsistent, or unbalanced way.\n"
            "Please respond with the following format exactly:\n"
            "<think>\n<your reasoning>\n</think>\n<answer>\nReal or Fake\n</answer>\n"
            "Here is the list of fashion products the user interacted with:\n{prompt_items}"
        )
        return input_template.format(prompt_items=prompt_items)

    if dataset == "MIND":
        prompt_items = "\n\n".join(
            [
                f"{i + 1}. category: {item.get('category', 'N/A')}\n"
                f"   Title: {item.get('title', 'N/A')}\n"
                for i, item in enumerate(items)
            ]
        )
        input_template = (
            "You are a careful and intelligent behavioral reviewer in a recommender system.\n"
            "Attackers may inject fake users with fabricated interaction histories to manipulate item rankings or degrade recommendation performance.\n"
            "Your task is to evaluate a user's interaction history and decide whether they are a **real, human user** or **a fake/malicious user**.\n"
            "Provide two things:\n"
            "1. A detailed explanation of your reasoning.\n"
            "2. A final judgment: 'Real' or 'Fake'.\n"
            "Guidelines:\n"
            "A real user typically has 2-3 prominent news topics they are particularly interested in, and there is usually some degree of coherence or connection between these topics. However, in the 'category' labels I provide, items labeled as 'news' may appear slightly more frequently than others, so you should be mindful of this when evaluating behavior.\n"
            "Cross-topic interaction is not inherently fake, but it becomes suspicious if the user's interactions with different topics appear scattered. If this occurs, you must appropriately lower your trust in the user being real.\n"
            "If a user has interacted with a wide variety of news categories but has not spent much time on any of them, you must appropriately lower your trust in them being real.\n"
            "You may also draw on your own knowledge and intuition to assist in determining whether the user's behavior resembles that of a genuine human.\n"
            "Please respond with the following format exactly:\n"
            "<think>\n<your reasoning>\n</think>\n<answer>\nReal or Fake\n</answer>\n"
            "Here is the list of news items the user interacted with:\n{prompt_items}"
        )
        return input_template.format(prompt_items=prompt_items)

    if dataset == "ml-1M":
        prompt_movies = "\n".join([f"{i + 1}. {item['name']} - {item['genres']}" for i, item in enumerate(items)])
        return (
            "You are a careful and intelligent behavioral reviewer in a recommender system.\n"
            "Attackers may inject fake users with fabricated interaction histories to manipulate item rankings or degrade recommendation performance.\n"
            "Your job is to assess a user based on their movie interaction history and determine whether they are a **real, normal user** or a **fake/malicious user**.\n"
            "You must provide two things:\n"
            "1. A detailed explanation of your reasoning.\n"
            "2. A final judgment: 'Real' or 'Fake'.\n"
            "**Guidelines for your judgment:**\n"
            "A real user typically has at least one clear area of interest, which is the most important criterion for judgment. In addition, there is a large number of movies in the drama and comedy genres. As a result, these two genres will also appear many times in the interaction history of fake users. Therefore, you must reduce the focus on these two categories!\n"
            "In this movie recommendation system, there are only 18 movie genres in total, so if a user interacts with almost all 18 genres, trust should be lowered accordingly!\n"
            "Please respond with the following format exactly:\n"
            "<think>\n<your reasoning>\n</think>\n<answer>\nReal or Fake\n</answer>\n"
            f"Here is the list of movies the user interacted with:\n{prompt_movies}"
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


__all__ = ["build_prompt"]


