import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field

@dataclass
class GenscanOutput:
    status: str = ""
    cds_list: list = field(default_factory=list) 
    intron_list: list = field(default_factory=list) 
    exon_list: list = field(default_factory=list) 

        
    @staticmethod
    def __get_cds(soup):
        cds_text = soup.find("pre").text
        cds_text = cds_text.split("Predicted peptide sequence(s):")[-1]
        cds_text = cds_text.strip().replace("\n\n", "\n")
        cds_list_unparsed = cds_text.split(">")[1:]
        cds_list = []
        for cds in cds_list_unparsed:
            cd = cds.split("\n")[1:]
            cd = ''.join(cd)
            cds_list.append(cd)
        return cds_list


    @staticmethod
    def __get_exons_introns(soup):
        exon_intron_data = soup.find("pre").text

        exon_intron_data = exon_intron_data.split("Predicted genes/exons:")[-1]
        exon_intron_data = exon_intron_data.split("Suboptimal exons with probability")[0]

        table_str = exon_intron_data.replace("\n\n\n\n", "\n").replace("\n\n\n", "\n").replace("\n\n", "\n")
        table_exon = []
        table_intron = []

        prev_exon_row = table_str.split("\n")[3].split()
        prev_exon = [prev_exon_row[0], int(prev_exon_row[3]), int(prev_exon_row[4])]
        table_exon.append(prev_exon)
        for i, row in enumerate(table_str.split("\n")[4:]):
            row_data = row.split()
            if len(row_data):
                index = row_data[0]
                begin = int(row_data[3])
                end = int(row_data[4])
                table_exon.append([index, begin, end])
                intron_begin = max(prev_exon[1], prev_exon[2]) + 1
                intron_end = min(begin, end) - 1
                table_intron.append([i, intron_begin, intron_end])
                prev_exon = [index, begin, end]
        return table_exon, table_intron
        
        
    def run_genscan(self, sequence=None, sequence_file=None, organism="Vertebrate", exon_cutoff=1.00, sequence_name=""):
    
        base_url = "http://hollywood.mit.edu"
        get_url = base_url + "/GENSCAN.html"
        response = requests.get(get_url)
        self.status = response.status_code

        soup_get = BeautifulSoup(response.text, "html")
        post_url = base_url + soup_get.find("form").attrs["action"]
        if sequence_file is not None:
            with open(sequence_file, 'rb') as f:
                sequence_file = f.read()
        payload = {
        "-o": organism,
        "-e": exon_cutoff,
        "-n": sequence_name,
        "-p": "Predicted peptides only",
        "-u": sequence_file,
        "-s": sequence
        }
        
        post_response = requests.post(post_url, data = payload)
        soup_post = BeautifulSoup(post_response.content, "lxml")
        self.exon_list, self.intron_list = self.__get_exons_introns(soup_post)
        self.cds_list = self.__get_cds(soup_post)
        