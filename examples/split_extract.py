"""Cut an email thread into emails, then pull fields out of each. Run: uv run python examples/split_extract.py"""

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline

from llama_index.wity import WityExtractor, WitySplitter

THREAD = """From: dana@example.com
The seal test on batch 2231 failed at 40 psi. Can we rerun it before Friday?

From: marcus@example.com
Yes. I booked the rig for Thursday 9 October at 10:00. It takes about 3 hours.
"""

pipeline = IngestionPipeline(
    transformations=[
        WitySplitter(split="one item per email"),
        WityExtractor(
            fields={"sender": "The email address the email is from", "when": "A date the email mentions"},
            types={"when": "date"},
        ),
    ]
)

for node in pipeline.run(documents=[Document(text=THREAD)]):
    print(node.metadata)
