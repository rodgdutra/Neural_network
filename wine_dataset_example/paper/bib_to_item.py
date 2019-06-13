# filename: bibtex2item.py
import sys

bibtex = """ @inproceedings{freire2018analysis,
  title={Analysis of Controlled Packet Departure to Support Ethernet Fronthaul Synchronization via PTP},
  author={Freire, Igor and Sousa, Ilan and Bemerguy, Pedro and Klautau, Aldebaro and Almeida, Igor and Lu, Chenguang and Berg, Miguel},
  booktitle={2018 IEEE International Symposium on Precision Clock Synchronization for Measurement, Control, and Communication (ISPCS)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}
"""
r = bibtex.split('\n')
i = 0
while i < len(r):
  line = r[i].strip()
  if not line: i += 1
  if '@' == line[0]:
    code = line.split('{')[-1][:-1]
    title = venue = volume = number = pages = year = publisher = authors = None
    output_authors = []
    i += 1
    while i < len(r) and '@' not in r[i]:
      line = r[i].strip()
      #print(line)
      if line.startswith("title"):
        title = line.split('{')[-1][:-2]
      elif line.startswith("journal"):
        venue = line.split('{')[-1][:-2]
      elif line.startswith("volume"):
        volume = line.split('{')[-1][:-2]
      elif line.startswith("number"):
        number = line.split('{')[-1][:-2]
      elif line.startswith("pages"):
        pages = line.split('{')[-1][:-2]
      elif line.startswith("year"):
        year = line.split('{')[-1][:-2]
      elif line.startswith("publisher"):
        publisher = line.split('{')[-1][:-2]
      elif line.startswith("author"):
        authors = line[line.find("{")+1:line.rfind("}")]
        for LastFirst in authors.split('and'):
          lf = LastFirst.replace(' ', '').split(',')
          if len(lf) != 2: continue
          last, first = lf[0], lf[1]
          output_authors.append("{}, {}.".format(last.capitalize(), first.capitalize()[0]))
      i += 1

    print "\\bibitem{%s}" % code
    if len(output_authors) == 1:
      print output_authors[0] + " {}. ".format(title),
    else:
      print ", ".join(_ for _ in output_authors[:-1]) + " & " + output_authors[-1] + " {}. ".format(title),
    if venue:
      print "{{\\em {}}}.".format(" ".join([_.capitalize() for _ in venue.split(' ')])),
      if volume:
        sys.stdout.write(" \\textbf{{{}}}".format(volume))
      if pages:
        sys.stdout.write(", {}".format(pages) if number else " pp. {}".format(pages))
      if year:
        sys.stdout.write(" ({})".format(year))
    if publisher and not venue:
      print "({},{})".format(publisher, year)
    print
    print