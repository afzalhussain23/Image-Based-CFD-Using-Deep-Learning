import argparse


def gen_blockMeshDict(x_cord, y_cord):
    """
    Create a `blockMeshDict` file for the geometry
    """

    scale = 1
    z = 0.05
    x_orig = 0
    y_orig = 0
    x_max = 3
    y_max = 1
    x_cord = x_cord
    y_cord = y_cord
    x_cell = int(x_cord * 50)
    y_cell = int(y_cord * 50)


    # Open file
    f = open("blockMeshDict", "w")

    # Write file
    f.write("/*--------------------------------*- C++ -*----------------------------------*\ \n"
            "| =========                |                                                  |\n"
            "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox            |\n"
            "|  \\    /   O peration     | Version:  5                                      |\n"
            "|   \\  /    A nd           | Web:      www.OpenFOAM.org                       |\n"
            "|    \\/     M anipulation  |                                                  |\n"
            "\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n"
            "{\n"
            "   version     2.0;\n"
            "   format      ascii;\n"
            "   class       dictionary;\n"
            "   object      blockMeshDict;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n")
    f.write("convertToMeters {};\n".format(scale))
    f.write("\n"
            "vertices\n"
            "("
            "\n")
    f.write("    ({} {} {})\n".format(x_orig, y_orig, -z))
    f.write("    ({} {} {})\n".format(x_cord, y_orig, -z))
    f.write("    ({} {} {})\n".format(x_orig, y_cord, -z))
    f.write("    ({} {} {})\n".format(x_cord, y_cord, -z))
    f.write("    ({} {} {})\n".format(x_max, y_cord, -z))
    f.write("    ({} {} {})\n".format(x_orig, y_max, -z))
    f.write("    ({} {} {})\n".format(x_cord, y_max, -z))
    f.write("    ({} {} {})\n".format(x_max, y_max, -z))
    f.write("    ({} {} {})\n".format(x_orig, y_orig, z))
    f.write("    ({} {} {})\n".format(x_cord, y_orig, z))
    f.write("    ({} {} {})\n".format(x_orig, y_cord, z))
    f.write("    ({} {} {})\n".format(x_cord, y_cord, z))
    f.write("    ({} {} {})\n".format(x_max, y_cord, z))
    f.write("    ({} {} {})\n".format(x_orig, y_max, z))
    f.write("    ({} {} {})\n".format(x_cord, y_max, z))
    f.write("    ({} {} {})\n".format(x_max, y_max, z))
    f.write(");\n"
            "\n"
            "blocks\n"
            "(\n")
    f.write("    hex (0 1 3 2 8 9 11 10) ({} {} {}) simpleGrading (1 1 1)\n".format(x_cell, y_cell, 1))
    f.write("    hex (2 3 6 5 10 11 14 13) ({} {} {}) simpleGrading (1 1 1)\n".format(x_cell, 50 - y_cell, 1))
    f.write("    hex (3 4 7 6 11 12 15 14) ({} {} {}) simpleGrading (1 1 1)\n".format(150 - x_cell, 50 - y_cell, 1))
    f.write(");\n"
            "\n"
            "edges\n"
            "(\n"
            ");\n"
            "\n"
            "boundary\n"
            "(\n"
            "    inlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (0 8 10 2)\n"
            "            (2 10 13 5)\n"
            "        );\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (4 7 15 12)\n"
            "        );\n"
            "    }\n"
            "    bottom\n"
            "    {\n"
            "        type symmetryPlane;\n"
            "        faces\n"
            "        (\n"
            "            (0 1 9 8)\n"
            "        );\n"
            "    }\n"
            "    top\n"
            "    {\n"
            "        type symmetryPlane;\n"
            "        faces\n"
            "        (\n"
            "            (5 13 14 6)\n"
            "            (6 14 15 7)\n"
            "        );\n"
            "    }\n"
            "    obstacle\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (1 3 11 9)\n"
            "            (3 4 12 11)\n"
            "        );\n"
            "    }\n"
            ");\n"
            "\n"
            "mergePatchPairs\n"
            "(\n"
            ");\n"
            "\n"
            "// ************************************************************************* //\n")

    # Close file
    f.close()


def writeCellInformation(x_cord, y_cord):
    x_cell = int(x_cord * 50)
    y_cell = int(y_cord * 50)

    f = open("cellInformation", "w")

    f.write("{} {}\n".format(x_cell, y_cell))
    f.write("{} {}\n".format(x_cell, 50 - y_cell))
    f.write("{} {}\n".format(150 - x_cell, 50 - y_cell))

    f.close()


# Total cell 7500

parser = argparse.ArgumentParser(description="Generating blockMeshDict file for the geometry")
parser.add_argument("x_cord", help="X coordinate of forward step")
parser.add_argument("y_cord", help="Y coordinate of forward step")
args = parser.parse_args()
gen_blockMeshDict(float(args.x_cord), float(args.y_cord))
writeCellInformation(float(args.x_cord), float(args.y_cord))

