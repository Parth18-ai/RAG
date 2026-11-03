import chromadb

-
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

client = chromadb.PersistentClient(path="./chroma_db")  # saved on disk
collection = client.get_or_create_collection(
    name="day16_chunks",
    metadata={"hnsw:space": "cosine"},  # use cosine, like in the numpy version
)

ids, documents, metadatas = [], [], []
for source_file, chunks in DOCS.items():
    for i, chunk in enumerate(chunks):
        ids.append(f"{source_file}_{i}")          # unique id per chunk
        documents.append(chunk)                   # the original text
        metadatas.append({"source": source_file, "chunk_no": i})  # sticky note

collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
print(f"Stored {collection.count()} chunks\n")

def search(query, top_k=3, source=None):
    """Semantic search. Optionally only look inside one source file."""
    where = {"source": source} if source else None

    results = collection.query(
        query_texts=[query],   # Chroma turns the question into coordinates
        n_results=top_k,
        where=where,           # metadata filter (None = search everything)
    )

    label = f" [only in {source}]" if source else ""
    print(f"Q: {query}{label}")
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  distance={dist:.3f} | {meta['source']} | {text}")
    print()

    question = input("Ask a question (or type 'quit'): ")
    if question.lower() == "quit":
        break

    file_name = input("Search only one file? Type its name or press Enter for all: ")
    search(question, source=file_name if file_name else None)
