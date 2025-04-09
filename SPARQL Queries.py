from rdflib import Graph
import pandas as pd

# Load the ontology
g = Graph()
file_path = "Eurostat KG.ttl"  # Update based on the uploaded file
g.parse(file_path, format="turtle")

# Define SPARQL queries
queries = {
    "ontology_metadata": """
        PREFIX dc: <http://purl.org/dc/terms/>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT ?title ?creator ?version ?description
        WHERE {
          OPTIONAL { <https://ec.europa.eu/eurostat/NLP4StatRef/ontology/> dc:title ?title. }
          OPTIONAL { <https://ec.europa.eu/eurostat/NLP4StatRef/ontology/> dc:creator ?creator. }
          OPTIONAL { <https://ec.europa.eu/eurostat/NLP4StatRef/ontology/> owl:versionInfo ?version. }
          OPTIONAL { <https://ec.europa.eu/eurostat/NLP4StatRef/ontology/> dc:description ?description. }
        }
    """,
    "class_count": """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT (COUNT(?class) AS ?totalClasses)
        WHERE { ?class a owl:Class. }
    """,
    "object_property_count": """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT (COUNT(?property) AS ?totalObjectProperties)
        WHERE { ?property a owl:ObjectProperty. }
    """,
    "datatype_property_count": """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT (COUNT(?property) AS ?totalDatatypeProperties)
        WHERE { ?property a owl:DatatypeProperty. }
    """,
    "list_classes": """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT ?class ?label
        WHERE {
          ?class a owl:Class.
          OPTIONAL { ?class rdfs:label ?label. }
        }
        ORDER BY ?class
        LIMIT 20  # Show only 20 classes
    """,
    "list_properties": """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT ?property ?type
        WHERE {
          { ?property a owl:ObjectProperty. BIND("ObjectProperty" AS ?type) }
          UNION
          { ?property a owl:DatatypeProperty. BIND("DatatypeProperty" AS ?type) }
        }
        ORDER BY ?property
        LIMIT 20  # Show only 20 properties
    """
}

# Function to execute queries
def run_sparql_query(query):
    results = g.query(query)
    data = [[str(val) for val in row] for row in results]
    return pd.DataFrame(data, columns=[var.replace("?", "") for var in results.vars])

# Store results
results_dict = {}

for query_name, query in queries.items():
    df = run_sparql_query(query)
    results_dict[query_name] = df

# Display summarized results
summary = {
    "Total Classes": results_dict["class_count"].iloc[0, 0],
    "Total Object Properties": results_dict["object_property_count"].iloc[0, 0],
    "Total Datatype Properties": results_dict["datatype_property_count"].iloc[0, 0]
}

print("🔹 **Ontology Summary** 🔹")
print(pd.DataFrame(summary, index=["Count"]))

print("\n🔹 **Ontology Metadata** 🔹")
print(results_dict["ontology_metadata"])

print("\n🔹 **Sample Classes (20 max)** 🔹")
print(results_dict["list_classes"])

print("\n🔹 **Sample Properties (20 max)** 🔹")
print(results_dict["list_properties"])

# Export full results to CSV
full_results = pd.concat(results_dict.values(), keys=results_dict.keys())
csv_path = "ontology_results.csv"
full_results.to_csv(csv_path, index=False)

print(f"\n✅ Full results saved to: {csv_path}")
