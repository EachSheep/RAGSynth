#!/bin/bash

python src/statistics.py --source_folder docs/admission.stanford.edu.new
# Total number of files: 65
# Total number of tokens: 132696
# Average tokens per file: 2041.4769230769232
# Maximum tokens in a single file: 7258 (File: docs/admission.stanford.edu.new/pdf/stanford_viewbook.pdf.md)

python src/statistics.py --source_folder docs/berkeley.edu.admission.new
# Total number of files: 281
# Total number of tokens: 1692134
# Average tokens per file: 6021.8291814946615
# Maximum tokens in a single file: 100554 (File: docs/berkeley.edu.admission.new/sites/default/files/uc_berkeley_asfsr.pdf.md

python src/statistics.py --source_folder docs/docs.cyotek.com.new
# Total number of files: 3560
# Total number of tokens: 2158768
# Average tokens per file: 606.3955056179775
# Maximum tokens in a single file: 17736 (File: docs/docs.cyotek.com.new/cyowcopy/1.9/licenses.html.md)

python src/statistics.py --source_folder docs/www.notion.so.help.new
# Total number of files: 477
# Total number of tokens: 2961885
# Average tokens per file: 6209.402515723271
# Maximum tokens in a single file: 17246 (File: docs/www.notion.so.help.new/help/guides.html.md)

python src/statistics.py --source_folder docs/hearthstone.fandom.com.wiki.new
# Total number of files: 22569
# Total number of tokens: 41652565
# Average tokens per file: 1845.5653772874296
# Maximum tokens in a single file: 132191 (File: hearthstone.fandom.com.wiki.new/wiki/Advanced_rulebook.html.md)
# Get-ChildItem -Path "hearthstone.fandom.com.wiki/" -Recurse -Filter "UserProfile*" | Remove-Item -Force
# Get-ChildItem -Path "hearthstone.fandom.com.wiki/" -Recurse -Filter "Mercenaries_full_art.html*" | Remove-Item -Force

python src/statistics.py --source_folder docs/zelda.fandom.com.wiki.new
# Total number of files: 11453
# Total number of tokens: 55414735
# Average tokens per file: 4838.447131755872
# Maximum tokens in a single file: 117432 (File: zelda.fandom.com.wiki.new/wiki/Link.html.md)

python src/statistics.py --source_folder docs/www.drugs.com.new
# Total number of files: 77552
# Total number of tokens: 282509162
# Average tokens per file: 3642.8352847121932
# Maximum tokens in a single file: 151116 (File: docs/www.drugs.com.new/pro/keytruda.html.md)

python src/statistics.py --source_folder docs/www.mayoclinic.org.new
# Total number of files: 2547
# Total number of tokens: 7662711
# Average tokens per file: 3008.5241460541815
# Maximum tokens in a single file: 30111 (File: www.mayoclinic.org.new/medical-professionals/cancer/news.html.md)
# Get-ChildItem -Path "cyowcopy/www.mayoclinic.org/" -Recurse -Filter "zh-hans" | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
# Get-ChildItem -Path "cyowcopy/www.mayoclinic.org/" -Recurse -Filter "ar" | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
# Get-ChildItem -Path "cyowcopy/www.mayoclinic.org/" -Recurse -Filter "ar.html" | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
# Get-ChildItem -Path "cyowcopy/www.mayoclinic.org/" -Recurse -Filter "es" | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
# Get-ChildItem -Path "cyowcopy/www.mayoclinic.org/" -Recurse -Filter "es.html" | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }