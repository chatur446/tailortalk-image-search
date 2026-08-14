from metadata_parser import parse_product_name


TEST_PRODUCTS = [
    "Pashmina - Banarasi Saree - Pink Colour",
    "Georgette Saree Magenta & Deep Purple",
    "Silk Saree Peach With Golden Zari Border And Subtle Floral Embroidery Border",
    "Pure Handloom Tussar Saree Dusty Rose",
    "Multicolour Abstract Print Saree with Mirror Work Blouse",
]


def main():

    for product in TEST_PRODUCTS:

        result = parse_product_name(product)

        print("\nProduct:")
        print(product)

        print("Fabrics:", result["fabrics"])
        print("Colours:", result["colours"])
        print("Designs:", result["designs"])


if __name__ == "__main__":
    main()