
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import RetrievalQA


DOCS = {
    "oracle_faq.txt": [
        "Oracle refunds are processed within 7 working days.",
        "Oracle database licences are renewed every year.",
        "Oracle support tickets are answered within 4 hours.",
        "To reset an Oracle cloud password, use the self-service portal.",
        "Oracle invoices are sent on the first day of each month.",
        "Oracle cloud backups run automatically every night.",
        "Oracle offers a 30 day free trial for new cloud accounts.",
        "Oracle consultants can be booked through the account manager.",
        "Oracle data centres are available in Mumbai and Hyderabad.",
        "Oracle contract changes need written approval from both sides.",
    ],
    "microsoft_faq.txt": [
        "Microsoft refunds take up to 14 working days.",
        "Microsoft 365 licences can be added or removed any time.",
        "Microsoft support is available 24 hours a day on chat.",
        "To reset an Azure password, click 'Forgot password' on the login page.",
        "Microsoft invoices are sent on the 15th of each month.",
        "Azure backups can be scheduled daily, weekly or monthly.",
        "Microsoft offers student discounts on selected products.",
        "Teams meetings can be recorded and saved to the cloud.",
        "Azure has data centres in Pune and Chennai.",
        "Microsoft partner agreements are reviewed every two years.",
    ],
    "hr_policy.txt": [
        "Employees get 24 days of paid leave every year.",
        "Sick leave needs a doctor's note after 3 days.",
        "Working hours are 9 AM to 6 PM, Monday to Friday.",
        "Work from home is allowed two days per week.",
        "New joiners complete a one week induction programme.",
        "Salary is credited on the last working day of the month.",
        "Performance reviews happen twice a year.",
        "Maternity leave is 26 weeks as per company policy.",
        "Employees can claim travel expenses within 30 days.",
        "The office is closed on public holidays and Sundays.",
    ],
    "it_support.txt": [
        "To fix a slow laptop, restart it and close unused apps.",
        "VPN access must be requested through the IT helpdesk.",
        "Lost laptops must be reported to IT within 24 hours.",
        "Passwords must be changed every 90 days.",
        "Use the company Wi-Fi named 'Excelencia-Secure' in the office.",
        "Software installation needs approval from the IT team.",
        "Printers are available on every floor near the pantry.",
        "Two-factor authentication is required for all email accounts.",
        "Shared drive files are backed up every night.",
        "Report phishing emails to the security team immediately.",
    ],
    "billing_guide.txt": [
        "Late payments attract a 2 percent monthly fee.",
        "Invoices can be paid by bank transfer or credit card.",
        "Credit notes are issued when a refund is approved.",
        "Payment terms for most clients are 30 days.",
        "GST is added to every invoice as per government rules.",
        "Billing disputes must be raised within 15 days of the invoice.",
        "Advance payments are adjusted against the next invoice.",
        "Monthly billing reports are shared with the finance team.",
        "Currency for international clients is US dollars.",
        "Duplicate invoices should be cancelled and reissued.",
    ],
}


if not os.path.exists("docs"):
    os.makedirs("docs")
    for file_name, lines in DOCS.items():
        with open(os.path.join("docs", file_name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))  
    print("Created docs/ folder")


loader = DirectoryLoader(
    "docs",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},  
)
documents = loader.load()                  
print(f"Loaded {len(documents)} files")


splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
chunks = splitter.split_documents(documents)   
print(f"Split into {len(chunks)} chunks")


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="day16_langchain",
    persist_directory="./chroma_langchain",
    collection_metadata={"hnsw:space": "cosine"},
    ids=[f"chunk_{i}" for i in range(len(chunks))],   
)
print(f"Stored {vectorstore._collection.count()} chunks\n")


llm = ChatOllama(model="llama3.2", temperature=0)


def ask(question, top_k=3, source=None):
    """Answer a question. Optionally only look inside one file."""
    search_kwargs = {"k": top_k}
    if source:
        search_kwargs["filter"] = {"source": os.path.join("docs", source)}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
    )
    result = qa.invoke({"query": question})

    print(f"\nQ: {question}")
    print(f"A: {result['result']}")
    print("Sources used:")
    for doc in result["source_documents"]:
        print(f"  {doc.metadata['source']} | {doc.page_content}")
    print()


while True:
    question = input("Ask a question (or type 'quit'): ")
    if question.lower() == "quit":
        break
    file_name = input("Search only one file? Type its name or press Enter for all: ")
    ask(question, source=file_name if file_name else None)