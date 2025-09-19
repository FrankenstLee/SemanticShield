import os
import json
import glob
import argparse
from tqdm import tqdm
from transformers import pipeline


def build_prompt(dataset, items):
    if dataset == "Clothing":
        prompt_items = '\n\n'.join([
            f"{i+1}. Title: {item.get('title', 'N/A')}\n"
            f"   Categories: {', '.join([cat.replace('Clothing, Shoes & Jewelry', '').strip() for cat in item.get('categories', []) if isinstance(cat, str)])}"
            for i, item in enumerate(items)
        ])
        input_template = (
            "You are a careful and intelligent behavioral reviewer in a recommender system.\n"
            "In this system, attackers might inject fake users with fabricated interaction histories to manipulate item rankings or degrade recommendation performance.\n"
            "Your task is to evaluate a user's interaction history and decide whether they are a **real, human user** or a **fake/malicious user**.\n"
            "Provide two things:\n"
            "1. A detailed explanation of your reasoning.\n"
            "2. A final judgment: 'Real' or 'Fake'.\n"
            "Guidelines:\n"
            "In the Clothing, Shoes & Jewelry domain, real users often show consistent behavior — such as a strong preference for one gender’s products, or logical patterns (e.g. dress + heels + bag).\n"
            "Users who interact mostly with one category, brand, or gender are typically genuine. If a user meets this criterion, you should consider them real.\n"
            "Cross-gender activity is not inherently fake, but it becomes suspicious if the user interacts with both male and female products in a scattered, inconsistent, or unbalanced way.\n"
            "Please respond with the following format exactly:\n"
            "<think>\n<your reasoning>\n</think>\n<answer>\nReal or Fake\n</answer>"
            "\nHere is the list of fashion products the user interacted with:\n{prompt_items}"
        )
        return input_template.format(prompt_items=prompt_items)

    elif dataset == "MIND":
        prompt_items = '\n\n'.join([
            f"{i+1}. category: {item.get('category', 'N/A')}\n"
            f"   Title: {item.get('title', 'N/A')}\n"
            for i, item in enumerate(items)
        ])
        input_template = (
            "You are a careful and intelligent behavioral reviewer in a recommender system.\n"
            "In this system, attackers might inject fake users with fabricated interaction histories to manipulate item rankings or degrade recommendation performance.\n"
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
            "<think>\n<your reasoning>\n</think>\n<answer>\nReal or Fake\n</answer>"
            "\nHere is the list of news items the user interacted with:\n{prompt_items}"
        )
        return input_template.format(prompt_items=prompt_items)

    elif dataset == "ml-1M":
        prompt_movies = '\n'.join([
            f"{i+1}. {m['name']} - {m['genres']}" for i, m in enumerate(items)
        ])
        prompt = (
            "You are a careful and intelligent behavioral reviewer in a recommender system.\n"
            "In this system, attackers might inject fake users with fabricated interaction histories to manipulate item rankings or degrade recommendation performance.\n"
            "Your job is to assess a user based on their movie interaction history and determine whether they are a **real, normal user** or a **fake/malicious user**.\n"
            "You must provide two things:\n"
            "1. A detailed explanation of your reasoning.\n"
            "2. A final judgment: 'Real' or 'Fake'.\n"
            "**Guidelines for your judgment:**\n"
            "A real user typically has at least one clear area of interest, which is the most important criterion for judgment. In addition, there is a large number of movies in the drama and comedy genres. As a result, these two genres will also appear many times in the interaction history of fake users. Therefore, you must reduce the focus on these two categories!\n"
            "In this movie recommendation system, there are only 18 movie genres in total, so if a user interacts with almost all 18 genres, trust should be lowered accordingly!\n"
            "Please respond with the following format exactly:\n"
            "<think>\n<your reasoning>\n</think>\n<answer>\nReal or Fake\n</answer>"
            f"Here is the list of movies the user interacted with:\n{prompt_movies}"
        )
        return prompt

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["Clothing", "MIND", "ml-1M"],
                        help="Which dataset format to use")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to input JSON files")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--device", type=int, default=0, help="GPU id (or -1 for CPU)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    generator = pipeline(
        "text-generation",
        model=args.model_path,
        device=args.device
    )

    files = glob.glob(f'{args.data_dir}/*.json')

    for file in tqdm(files, desc='Processing files'):
        filename = os.path.basename(file)
        file_stem = os.path.splitext(filename)[0]
        out_path = os.path.join(args.out_dir, file_stem + '.txt')

        with open(file, 'r') as f:
            data = json.load(f)

        with open(out_path, 'w') as fout:
            for user_id, items in data.items():
                prompt = build_prompt(args.dataset, items)

                try:
                    res = generator(
                        [{"role": "user", "content": prompt}],
                        max_new_tokens=512,
                        temperature=0.1,
                        top_p=0.9,
                        top_k=50,
                        return_full_text=False
                    )[0]["generated_text"].strip()
                except Exception as e:
                    res = f"Error: {e}"

                print(f"\nUser: {user_id}")
                print(res)
                print("=" * 60)

                fout.write(f"User: {user_id}\n\n")
                fout.write(res + "\n")
                fout.write("=" * 60 + "\n")
                fout.flush()
                os.fsync(fout.fileno())


if __name__ == "__main__":
    main()
