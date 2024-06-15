import os
import typing as tp
from pdfminer.high_level import extract_pages, extract_text_to_fp
import xml.etree.ElementTree as ET
import re
import tqdm
import pandas as pd
import numpy as np

from retriever.embedder import Embedder


def list_files_with_ext(
    path: str,
    ext: str
    ) -> tp.List[str]:
    """
        List files in folder and choose ones with pdf extension
    """
    result = []
    for file in os.listdir(path):
        if file.endswith(ext):
            # gather full path
            result.append(os.path.join(path, file))
    return result


def pdf_to_xml(
    path: str,
    output_dir: str
    ) -> None:
    """
        Transform PDF's into XML's
    """
    output = os.path.join(output_dir, os.path.basename(path).replace(".pdf", ".xml"))
    output_path = os.path.join(output_dir, os.path.basename(output)).replace(".xml", "")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(path, 'rb') as f:
        extract_text_to_fp(inf=f, outfp=open(output, 'wb'), output_dir=output_path, layout=False, output_type='xml', codec='utf-8')


def process_xml(
    path: str,
    word_split: int = -1,
    long_text_fallback: int = 2000
    ) -> tp.List[tp.Any]:
    """
        Parse XML's into separate sub-documents (index)
    """
    basename = os.path.basename(path).replace(".xml", "")
    # open xml file read into tree
    tree = ET.parse(path)
    root = tree.getroot()
    # gather all <page> elements
    pages = root.findall(".page")
    last_text = None
    elem_list = []
    page_index = 0
    for page in pages:
        for element in page.findall(".//*"):
            if element.tag == "text":
                # remove all keys except size and font
                element.attrib = {"size":element.attrib["size"], "page": str(page_index)}
                if last_text is None:
                    last_text = element
                elif float(element.attrib["size"]) == float(last_text.attrib["size"]):
                    last_text.text += element.text
                else:
                    if (len(last_text.text)) > 2:
                        elem_list.append(last_text)
                    last_text = element
            if element.tag == "image" and int(element.attrib["width"]) > 150 and int(element.attrib["height"]) > 100:
                if last_text is not None and (len(last_text.text)) > 2:
                    elem_list.append(last_text)
                last_text = None
                elem_list.append(element)
        page_index += 1
    if last_text is not None:
        if (len(last_text.text)) > 2:
            elem_list.append(last_text)
    # gather texts counts
    text_counts = {}
    for element in elem_list:
        if element.tag == "text":
            if element.text in text_counts:
                text_counts[element.text] += 1
            else:
                text_counts[element.text] = 1
    # drop all elements with more than 2 occurrences from element list. Also remove ones containing "страница" and "страницы" simultaneously in text
    filtered_elem_list = []
    chapter_names = ""
    for element in elem_list:
        if element.tag == "text":
            if text_counts[element.text] > 2:
                continue
            if element.text.lower().find("страница") != -1 and element.text.lower().find("страниц") != -1:
                continue
            if int(element.attrib["page"]) < 10 and (element.text.find("................") != -1 or element.text.find("……") != -1):
                chapter_names += element.text + " "
                continue
        filtered_elem_list.append(element)

    # gather text sizes and length of matching texts
    size_lengths = {}
    for e in filtered_elem_list:
        if e.tag == "text":
            if e.attrib["size"] not in size_lengths:
                size_lengths[e.attrib["size"]] = 0
            size_lengths[e.attrib["size"]] += len(e.text)

    # drop all elements with size less than the most frequent sizes
    elem_list = filtered_elem_list
    filtered_elem_list = []
    top_size = float(sorted(size_lengths.items(), key = lambda x : -x[1])[0][0])
    for element in elem_list:
        if element.tag == "text" and float(element.attrib["size"]) < top_size:
            continue
        filtered_elem_list.append(element)
    
    results_words = []
    
    if word_split > 0 or long_text_fallback > 0:
        actual_split = word_split
        if actual_split < 0:
            actual_split = 200
        # go through all elements, concatenating text and listing images, until length of the text gets above simpel_split. Then append data to results and start filling again
        text = ""
        images = []
        pages = set()
        idx = 0
        for element in filtered_elem_list:
            if element.tag == "image":
                images += [element.attrib["src"]]
            elif element.tag == "text":
                text += element.text
                pages.add(int(element.attrib["page"]))
                if len(text.split(' ')) > actual_split:
                    results_words.append({"document": str(basename), "id": str(basename) + "/" + str(idx), "text": text, "images": images, "pages": pages})
                    text = ""
                    images = []
                    pages = set()
                    idx += 1
        if len(text) > 0:
            results_words.append({"document": str(basename), "id": str(basename) + "/" + str(idx), "text": text, "images": images, "pages": pages})
        if word_split > 0:
            return results_words

    # replace sequence of more than one periods in chapter_names with as single | symbol
    chapter_names = re.sub(r"\.\.+", "|", chapter_names)
    # process strange three dots
    chapter_names = re.sub(r"\…+", "|", chapter_names)
    # remove all digits and periods from chapter names
    chapter_names = re.sub(r"([А-Я]\-)\d", "", chapter_names)
    chapter_names = re.sub(r"\d\.", "", chapter_names)
    chapter_names = re.sub(r"\.\d", "", chapter_names)
    chapter_names = re.sub(r"\|(\s*\d\s*)*", "|", chapter_names)
    # remove all spaces around | symbol
    chapter_names = re.sub(r"\s*\|+\s*", "|", chapter_names)
    chapter_names = [x for x in chapter_names.split("|") if len(x) > 5]

    contains_text = False
    text = ""
    images = []
    pages = set()
    idx = 0
    results = []
    for element in filtered_elem_list:
        if element.tag == "text":
            if float(element.attrib["size"]) > top_size + 0.5:
                if contains_text:
                    results.append({"document": str(basename), "id": str(basename) + "/" + str(idx), "text": text, "images": images, "pages": pages})
                    images = []
                    text = ""
                    pages = set()
                    idx += 1
                contains_text = False
            else:
                for c in chapter_names:
                    if element.text.find(c) != -1 and contains_text:
                        results.append({"document": str(basename), "id": str(basename) + "/" + str(idx), "text": text, "images": images, "pages": pages})
                        images = []
                        text = ""
                        pages = set()
                        idx += 1
                        break
                contains_text = True
            pages.add(int(element.attrib["page"]))
            text += element.text
        else:
            images += [element.attrib["src"]]
    if contains_text:
        results.append({"document": str(basename), "id": str(basename) + "/" + str(idx), "text": text, "images": images, "pages": pages})
    
    if long_text_fallback > 0:
        max_len = -1
        for r in results:
            max_len = max(max_len, len(r["text"]))
        if max_len > long_text_fallback:
            results = results_words
    return results

def embed_texts(
    texts: tp.Iterable
) -> tp.List[np.ndarray]:
    model = Embedder.from_resources_path(
        resources_path='cointegrated/rubert-tiny2',
        device='cpu'
    )
    responses = []
    for text in tqdm.tqdm(texts):
        resp = model(text)
        responses.append(resp)
    
    return responses

def main():
    in_dir = "../inputs"
    data_dir = "data"

    

    # convert each file to xml in data folder, also extract images from pdf
    files = list_files_with_ext(in_dir, ".pdf")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for file in tqdm.tqdm(files):
        pdf_to_xml(file, data_dir)

    # cleaup xml data and gather everything to csv
    files = list_files_with_ext(data_dir, ".xml")
    output = []
    for file in tqdm.tqdm(files):
        output += process_xml(file)

    # Build DataFrame
    df = pd.DataFrame(output, columns=["document", "id", "text", "images", "pages"])

    # Embed texts
    embeddings = embed_texts(df['text'].values)
    df['embeddings'] = embeddings

    # Save to CSV
    df.to_csv("data/database.csv", index=False)

if __name__ == "__main__":
    main()