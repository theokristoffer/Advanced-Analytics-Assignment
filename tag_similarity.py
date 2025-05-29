import json
from itertools import combinations

def jaccard_similarity(tags1, tags2):
    set1 = set(tags1)
    set2 = set(tags2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def main(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # extract apps and metadata
    apps = []
    for node in data.get("nodes", []):
        if "App" in node.get("labels", []):
            props = node.get("properties", {})
            name = props.get("name", "Unknown")
            tags = props.get("tags", [])
            apps.append({"name": name, "tags": tags})

    # print app names and tags
    # print("Apps and tags:")
    # for app in apps:
    #     print(f"{app['name']}: {app['tags']}")

    # compute similarity between apps based on tags
    print("\nSimilarity Scores (Jaccard):")
    for (app1, app2) in combinations(apps, 2):
        sim = jaccard_similarity(app1["tags"], app2["tags"])
        print(f"{app1['name']} <-> {app2['name']}: {sim:.3f}")

if __name__ == "__main__":
    json_file = "C:\\Users\\Theo\\Downloads\\community_mining.json"
    main(json_file)
