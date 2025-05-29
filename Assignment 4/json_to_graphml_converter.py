import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

def json_to_graphml(json_file_path, output_file_path):
    """
    Convert JSON graph data to GraphML format.
    
    Args:
        json_file_path (str): Path to input JSON file
        output_file_path (str): Path to output GraphML file
    """
    
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create GraphML root element
    graphml = ET.Element('graphml')
    graphml.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
    graphml.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    graphml.set('xsi:schemaLocation', 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd')
    
    # Define attribute keys for nodes
    node_keys = {}
    
    # Define keys for User properties
    user_attrs = ['avatar', 'community_id', 'communityvisibilitystate', 'ident', 
                  'personaname', 'personastate', 'profilestate', 'profileurl', 
                  'realname', 'timecreated', 'owned_app_ids']
    
    for i, attr in enumerate(user_attrs):
        key_elem = ET.SubElement(graphml, 'key')
        key_elem.set('id', f'n{i}')
        key_elem.set('for', 'node')
        key_elem.set('attr.name', attr)
        if attr == 'owned_app_ids':
            key_elem.set('attr.type', 'string')  # Will store as comma-separated string
        elif attr in ['community_id', 'communityvisibilitystate', 'personastate', 'profilestate']:
            key_elem.set('attr.type', 'int')
        else:
            key_elem.set('attr.type', 'string')
        node_keys[attr] = f'n{i}'
    
    # Add node type key
    type_key = ET.SubElement(graphml, 'key')
    type_key.set('id', 'ntype')
    type_key.set('for', 'node')
    type_key.set('attr.name', 'type')
    type_key.set('attr.type', 'string')
    
    # Add node label key
    label_key = ET.SubElement(graphml, 'key')
    label_key.set('id', 'nlabel')
    label_key.set('for', 'node')
    label_key.set('attr.name', 'label')
    label_key.set('attr.type', 'string')
    
    # Define keys for edge properties
    playtime_key = ET.SubElement(graphml, 'key')
    playtime_key.set('id', 'e0')
    playtime_key.set('for', 'edge')
    playtime_key.set('attr.name', 'playtime')
    playtime_key.set('attr.type', 'int')
    
    # Create graph element
    graph = ET.SubElement(graphml, 'graph')
    graph.set('id', 'G')
    graph.set('edgedefault', 'directed')
    
    # Process nodes
    nodes_data = data.get('nodes', [])
    for node_data in nodes_data:
        node = ET.SubElement(graph, 'node')
        node.set('id', str(node_data['id']))
        
        # Add node type
        type_data = ET.SubElement(node, 'data')
        type_data.set('key', 'ntype')
        type_data.text = node_data.get('type', 'node')
        
        # Add node labels
        labels = node_data.get('labels', [])
        if labels:
            label_data = ET.SubElement(node, 'data')
            label_data.set('key', 'nlabel')
            label_data.text = ','.join(labels)
        
        # Add node properties
        properties = node_data.get('properties', {})
        for prop_name, prop_value in properties.items():
            if prop_name in node_keys:
                data_elem = ET.SubElement(node, 'data')
                data_elem.set('key', node_keys[prop_name])
                
                if prop_name == 'owned_app_ids' and isinstance(prop_value, list):
                    data_elem.text = ','.join(map(str, prop_value))
                else:
                    data_elem.text = str(prop_value)
    
    # Process relationships/edges
    relationships_data = data.get('relationships', [])
    for rel_data in relationships_data:
        edge = ET.SubElement(graph, 'edge')
        edge.set('id', str(rel_data['id']))
        edge.set('source', str(rel_data['start']))
        edge.set('target', str(rel_data['end']))
        edge.set('label', rel_data.get('label', ''))
        
        # Add edge properties
        properties = rel_data.get('properties', {})
        if 'playtime' in properties:
            playtime_data = ET.SubElement(edge, 'data')
            playtime_data.set('key', 'e0')
            playtime_data.text = str(properties['playtime'])
    
    # Convert to pretty-printed XML string
    xml_str = ET.tostring(graphml, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # Remove empty lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    # Write to file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"Successfully converted {json_file_path} to {output_file_path}")
    print(f"Processed {len(nodes_data)} nodes and {len(relationships_data)} relationships")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_json = "C:\\Users\\Theo\\Downloads\\graph3.json"
    output_graphml = "output_graph4.graphml"
    
    try:
        json_to_graphml(input_json, output_graphml)
    except FileNotFoundError:
        print(f"Error: Could not find the file {input_json}")
        print("Please make sure the file path is correct")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")
