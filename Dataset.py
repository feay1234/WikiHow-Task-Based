import pandas as pd
import os
if __name__ == '__main__':

    # crawl("how to cook pasta", "csv/")

    all_categories = "Arts and Entertainment·Cars & Other Vehicles·Computers and Electronics·Education and Communications·Family Life·Finance and Business·Food and Entertaining·Health·Hobbies and Crafts·Holidays and Traditions·Home and Garden·Personal Care and Style·Pets and Animals·Philosophy and Religion·Relationships·Sports and Fitness·Travel·Work World·Youth"
    all_categories = all_categories.lower()
    all_categories = all_categories.split("·")
    all_categories = ['sports and fitness']

    df = pd.read_csv("data/wikihowSep.csv")
    df = df[df.sectionLabel != "Steps"]
    df["title"] = df["title"].str.lower()
    df["sectionLabel"] = df["sectionLabel"].str.lower()

    wikiCat = pd.read_csv("data/cate.csv", sep=",", error_bad_lines=False, names=["title", "category"])
    wikiCat['title'] = wikiCat['title'].str.replace("https://www.wikihow.com/", "").str.replace("%22", "").str.replace(
        "-", " ").str.lower()
    wikiCat['title'] = ["how to " + i for i in wikiCat['title'].tolist()]
    wikiCat['category'] = wikiCat['category'].str.lower()

    for cat in all_categories[::-1]:
        print(cat)
        tasks = wikiCat[wikiCat.category.str.contains(cat)].title.tolist()

        cat = cat.replace(" ", "_")

        for d in ["csv", "csv2"]:

            print(d)
            folder = "%s/%s/" % (d, cat)
            if not os.path.exists(folder):
                continue
                # os.makedirs(folder)

            finish = []
            for file in os.listdir(folder):
                if os.stat(folder + file).st_size > 0:
                    if ".csv" in file:
                        finish.append(file.split(".csv")[0].replace("_", " "))

            error = []
            if os.path.exists(folder+cat):
                error.extend([line.rstrip('\n') for line in open(folder+cat)])
            finish.extend(error)


            # df["headline"] = df["headline"].str.lower().str.replace("\n", "").str.replace(".","")
            # df["title"] = df["title"].str.lower()
            # print(tasks)

            queries = []
            for name, row in df[df.title.isin(tasks)].drop_duplicates(["sectionLabel"]).groupby("title"):
                queries.append(name)
                queries.extend(row.sectionLabel.tolist())

            to_crawl = []
            for t in queries:
                if t not in finish:
                    to_crawl.append(t)


            print("Finished: %d, Error: %d, To crawl: %d" % (len(finish), len(error), len(to_crawl)))


