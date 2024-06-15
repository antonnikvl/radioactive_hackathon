import logging
from pathlib import Path

from embedder import Embedder

import flask
import torch
import numpy as np
import pandas as pd


logger = logging.getLogger('Embedder')

logger.info('Loading model...')

model = Embedder.from_resources_path('cointegrated/rubert-tiny2', device='cpu')
model.eval()

document_embeddings_df = pd.read_csv('documents.csv')
document_ids = document_embeddings_df['document_id'].values
document_embeddings = np.array([np.fromstring(emb[1:-1], sep=',') for emb in document_embeddings_df['embedding']])

logger.info('Model loaded')
app = flask.Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
	data = flask.request.json
	failed = False
	if data is None:
		logger.error('No data in request')
		failed = True
	elif 'Context' not in data:
		logger.error('No Context in request')
		failed = True
	elif not isinstance(data['Context'], list):
		logger.error('Context is not a list')
		failed = True
	elif len(data['Context']) != 1:
		logger.error('Context list is not of length 1')
		failed = True
	if failed:
		return flask.Response(status=400)
	text = data['Context'][0]

	print(text)

	text_emb = model(text)
	print(text_emb.shape, document_embeddings.shape)
	similarities = np.dot(document_embeddings, text_emb)
	print(similarities)
	print(np.argsort(similarities)[::-1])
	print(np.sort(similarities)[::-1])

	top_k = 5
	top_k_indices = np.argsort(similarities)[-top_k:][::-1]

	print(top_k_indices.tolist())

	top_k_doc_ids = [document_ids[i] for i in top_k_indices]

	for i in range(top_k):
		print(top_k_indices.tolist()[i], document_ids[top_k_indices.tolist()[i]])

	return flask.jsonify({'TopDocumentIds': top_k_doc_ids})


@app.route('/health', methods=['GET'])
def health():
    return flask.Response(status=200, response='ok')


@app.route('/update_db',  methods=['POST'])
def update():
	global document_embeddings_df
	global document_ids
	global document_embeddings

	data = flask.request.json
	failed = False
	if data is None:
		logger.error('No data in request')
		failed = True
	elif 'File' not in data:
		logger.error('No File in request')
		failed = True
	elif not isinstance(data['File'], list):
		logger.error('File is not a list')
		failed = True
	elif len(data['File']) != 1:
		logger.error('File list is not of length 1')
		failed = True
	if failed:
		return flask.Response(status=400)
	text = data['File'][0]

	document_embeddings_df = pd.read_csv(text)
	document_ids = document_embeddings_df['document_id'].values
	document_embeddings = np.array([np.fromstring(emb[1:-1], sep=',') for emb in document_embeddings_df['embedding']])

	return flask.Response(status=200, response='ok')


if __name__ == '__main__':
	app.run(debug=False)

