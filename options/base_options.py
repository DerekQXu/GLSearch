import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--base', required=True, type=str, help='base parameter')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
            
        self.opt = self.parser.parse_args()
                
        return self.opt
