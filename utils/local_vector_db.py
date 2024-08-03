"""
Create a local vector DB using Chroma

Author: fvilmos, https://github.com/fvilmos
"""
from langchain_community.document_loaders import WebBaseLoader, TextLoader,PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from contextlib import suppress
import os


class LocalVectorDB():
    """
    Uses Chroma to store embeddings in vecor db
    Has a safegurd option to do not add again embeddings.
    """
    def __init__(self, db_location:str, db_name:str) -> None:
        """
        initialize variables, embedding model

        Args:
            db_location (str): directory name to store the vectors
            db_name (str): 
        """
        self.db_location=db_location
        self.db_name=db_name
        self.db = None

        if os.path.exists(self.db_location)==False:
            # create db directory
            os.mkdir(self.db_location)
        
        oembed = OllamaEmbeddings(model="nomic-embed-text")
        self.db = Chroma(persist_directory=self.db_location, embedding_function=oembed, collection_metadata={"name":self.db_name})
    
    def _load_documents(self,doc_list:list[str]):
        """
        Load different type of documents
        Args:
            doc_list (list[str]): list of documents to be loaded, ie. ["G20.pdf","http://test.com"]

        Returns:
            _type_: list of document content
        """

        documents = []
        #guess doc type type
        for p in doc_list:
            doc = None
            if 'http' in p:
                print ("input type: http, ", p)
                try:
                    doc = WebBaseLoader(p).load()
                except BaseException as err:
                    print (f"Exception: {err}, input skipped!")
            elif '.txt' in p:
                print ("input type: txt, ", p)
                try:
                    doc = TextLoader(p, autodetect_encoding=True).load()
                except BaseException as err:
                    print (f"Exception: {err}, input skipped!")
            elif '.pdf' in p:
                print ("input type: pdf, ", p)
                try:
                    doc = PyPDFLoader(p).load()
                except BaseException as err:
                    print (f"Exception: {err}, input skipped!")
            else:
                # unsupported document, skipp it
                print ("input type: unknown, will be skippped!")
                pass
            if doc is not None:
                documents.append(doc)
        docs_list = [item for sublist in documents for item in sublist]

        return docs_list
    
    def add_to_vector_db(self, file_list:list[str], chunk_size=800, chunk_overlap=80):
        """
        Add list of files or urls to the vector db

        Args:
            file_list (list[str]): list of inputs i.e. ["G20.pdf"]
            chunk_size (int, optional): Minimum test slice. Defaults to 800.
            chunk_overlap (int, optional): character overlap in chunks. Defaults to 80.
        """

        # load documents
        docs = self._load_documents(file_list)

        # split a document in smaller parts
        splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        splits = splitter.split_documents(docs)

        # get current items
        current_items = self.db.get(include=[])
        current_items_ids = set(current_items['ids'])

        # update new information
        splits_ids = self._get_current_split_ids(splits)
        new_splits = []
        for s in splits_ids:
            if s.metadata['id'] not in current_items_ids:
                new_splits.append(s)
        
        if len(new_splits) > 0:
            # add new spilts to the exsiting ones
            new_split_ids = [ns.metadata["id"] for ns in new_splits]
            self.db.add_documents(new_splits, ids=new_split_ids)
        else:
            print ("No new data to add...")
        
        
    def _get_current_split_ids(self,splits):
        """
        Provide the IDs for the chunks

        Args:
            splits (Document): document parts

        Returns:
            str: IDs
        """

        splits_index = 0
        page_id = None
        
        for s in splits:
            source = s.metadata.get('source')
            l_page = s.metadata.get('page')
            
            # construct page id
            current_page_id = f'{source}:{l_page}'

            # test if exist
            if current_page_id == page_id:
                splits_index +=1
            else:
                splits_index = 0
            
            # update split ids
            splits_id = f'{current_page_id}:{splits_index}'
            s.metadata['id'] = splits_id
            page_id = current_page_id

        return splits
    
    def delete_db(self):
        """
        Remove local db
        """
        # check if db exit, then delete
        if os.path.exists(self.db_location):
            with suppress(OSError):
                os.remove(self.db_location)
