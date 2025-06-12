from tasks.segment_wound_area import segment_wound_area
from tasks.segment_wound_edge import segment_wound_edge
from helpers import load_config, save_image

config = load_config("/teamspace/studios/this_studio/config.yaml")

if __name__ == "__main__":
    # area = segment_wound_area(config)
    edge = segment_wound_edge(config)
    save_image(edge, "test.png")