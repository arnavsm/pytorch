# https://docs.python.org/3/library/argparse.html
import math
import argparse
from argparse import ArgumentParser

# Create an ArgumentParser object
parser = ArgumentParser(description='Calculate Volume of Cylinder')

# Positional Arguments
parser.add_argument('radius', type=int, help = 'Radius of Cylinder')
parser.add_argument('height', type=int, help = 'Height of Cylinder')

# Optional mutually exclusive arguments for quiet and verbose modes
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
group.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

# Parse the arguments
args = parser.parse_args()
### Usage: python argparse_practice.py 2 3


# Optional Arguments (try with reguires = True as well)
# parser.add_argument('--radius', type=int, help = 'Radius of Cylinder')


# Short Hand Notation
# parser.add_argument('-r', '--radius', type=int, help = 'Radius of Cylinder')
# parser.add_argument('-H', '--height', type=int, help = 'Height of Cylinder')
### Usage with optional args: python argparse_practice.py -r 2 -H 3


# Function to calculate the volume of a cylinder
def cyl_vol(radius, height):
    return (math.pi * (radius**2) * height)

if __name__ == '__main__':
    # Calculate the volume
    volume = cyl_vol(args.radius, args.height)

    # Output based on the mode
    if args.quiet:
        print(volume)
    elif args.verbose:
        print(f'The volume of a cylinder with radius {args.radius} and height {args.height} is {volume:.2f}')
    else:
        print(f'Volume: {volume:.2f}')