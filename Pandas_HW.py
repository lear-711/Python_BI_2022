
import gffpandas.gffpandas as gffpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 1.1

# ### Функция для чтения gff файлов:

def read_gff(file_gff_path):
    annotation = gffpd.read_gff3(file_gff_path)
    annotation_df = annotation.df
    return annotation_df


rrna_annotation = read_gff('rrna_annotation.gff')
rrna_annotation


# ### Функция для чтения bed файлов:

def read_bed6(file_bed_path):
    alingment = pd.read_csv(file_bed_path, sep = '\t', names = ['chromosome', 'start', 'end', 'name', 'score', 'strand'])
    return alingment


read_bed6('alignment.bed')


# # 1.2

rrna_annotation['attributes'] = rrna_annotation['attributes'].apply(lambda x: x.split('=')[2].split(' ')[0])
rrna_annotation


# # 1.3

count_rrna_df = rrna_annotation.groupby(['seq_id','attributes']).size().reset_index(name='Count')
count_rrna_df = count_rrna_df.rename(columns = {'seq_id':'Sequence', 'attributes':'RNA type'})
count_rrna_df


plt.subplots(figsize=(40, 20))
sns.barplot(x="Sequence", y="Count", hue = 'RNA type', data=count_rrna_df, width = 0.5)
plt.xticks(rotation=90, size = 20)
plt.yticks(size = 20)
plt.xlabel('Sequence', size=30)
plt.ylabel('Count', size=30)
plt.legend(title="RNA type", fontsize=25, title_fontsize=25)


# # 2

# ### Считываем данные:

diffexpr_data = pd.read_csv('diffexpr_data.tsv.gz', sep = '\t')
diffexpr_data


# ### Записываем в новый массив значения logFC, где p-value > 0.05:

logFC = diffexpr_data.query('log_pval > 0.05')['logFC']


# ### Находим значения logFC топ-2 генов, значимо снизивших экспрессию, и топ-2 генов, значимо увеличивших экспрессию:

logFC_sorted = sorted(logFC)

A_min = logFC_sorted[0]
B_min = logFC_sorted[1]
print(A_min, B_min)

A_max = logFC_sorted[-1]
B_max = logFC_sorted[-2]
print(A_max, B_max)


# ### Находим значения p-value, соответствующие полученным значениям logFC:

Ap_min = diffexpr_data[diffexpr_data['logFC'] == A_min]['log_pval']
A_pval_min = Ap_min.to_list()[0]

Bp_min = diffexpr_data[diffexpr_data['logFC'] == B_min]['log_pval']
B_pval_min = Bp_min.to_list()[0]

Ap_max = diffexpr_data[diffexpr_data['logFC'] == A_max]['log_pval']
A_pval_max = Ap_max.to_list()[0]

Bp_max = diffexpr_data[diffexpr_data['logFC'] == B_max]['log_pval']
B_pval_max = Bp_max.to_list()[0]


# ### Находим названия генов:

Ag_min = diffexpr_data[diffexpr_data['logFC'] == A_min]['Sample']
A_gene_min = Ag_min.to_list()[0]

Bg_min = diffexpr_data[diffexpr_data['logFC'] == B_min]['Sample']
B_gene_min = Bg_min.to_list()[0]

Ag_max = diffexpr_data[diffexpr_data['logFC'] == A_max]['Sample']
A_gene_max = Ag_max.to_list()[0]

Bg_max = diffexpr_data[diffexpr_data['logFC'] == B_max]['Sample']
B_gene_max = Bg_max.to_list()[0]


# ### Строим violent plot:

fig, ax = plt.subplots(figsize=(25, 15))
plt.rc('font')

down_left = diffexpr_data[(diffexpr_data['logFC']<=0)&(diffexpr_data['log_pval']<=0.05)]
up_left = diffexpr_data[(diffexpr_data['logFC']<=0)&(diffexpr_data['log_pval']>=0.05)]
down_right = diffexpr_data[(diffexpr_data['logFC']>=0)&(diffexpr_data['log_pval']<=0.05)]
up_right = diffexpr_data[(diffexpr_data['logFC']>=0)&(diffexpr_data['log_pval']>=0.05)]

plt.scatter(x=down_left['logFC'],y=down_left['log_pval'],s=20,label="Non-significantly downregulated",color="green")
plt.scatter(x=up_left['logFC'],y=up_left['log_pval'],s=20,label="Significantly downregulated",color="blue")
plt.scatter(x=down_right['logFC'],y=down_right['log_pval'],s=20,label="Non-significantly upregulated",color="red")
plt.scatter(x=up_right['logFC'],y=up_right['log_pval'],s=20,label="Significantly upregulated",color="orange")

# Устанавливаем название и параметры осей
#plt.rc('axes', linewidth=4)
plt.xlabel(r'$\bf{log_2}$ (fold change)', size = 25, fontweight = 'bold', style='italic')
plt.ylabel(r'$\bf{- log_{10}}$ (p-value corrected)', size = 25, fontweight = 'bold', style='italic')
plt.rcParams["axes.labelweight"] = "bold"

# Устанавливаем ticks
plt.xticks(size = 20, fontweight = 'bold')
plt.yticks(size = 20, fontweight = 'bold')
plt.minorticks_on()
plt.tick_params(axis="x", which="minor", length=5, width=3)
plt.tick_params(axis="y", which="minor", length=5, width=3)
plt.tick_params(axis="x", which="major", length=5, width=3)
plt.tick_params(axis="y", which="major", length=5, width=3)

# Название графика:
plt.title("Volcano plot", size = 40, fontweight = 'bold', style='italic')

# Пунктирные линии:
plt.axvline(0,color="grey",linestyle="--", linewidth = 3)
plt.axhline(0.05,color="grey",linestyle="--", linewidth = 3)
plt.text(8, 1.2, 'p-value = 0.05', fontsize = 20, fontweight = 'normal')

# Легенда
plt.legend(prop = {'weight':'bold', 'size':20}, markerscale=4, shadow=True)

# Аннотируем гены
ax.annotate(A_gene_min, xy = (A_min, A_pval_min), size = 25, xytext=(-10.5, 60), weight='bold',
            arrowprops=dict(arrowstyle='simple', fc='red', lw=1))
ax.annotate(B_gene_min, xy = (B_min, B_pval_min), size = 25, xytext=(-9, 10), weight='bold',
            arrowprops=dict(arrowstyle='simple', fc='red', lw=1))
ax.annotate(A_gene_max, xy = (A_max, A_pval_max), size = 25, xytext=(5, 5), weight='bold',
            arrowprops=dict(arrowstyle='simple', fc='red', lw=1))
ax.annotate(B_gene_max, xy = (B_max, B_pval_max), size = 25, xytext=(4, 10), weight='bold',
            arrowprops=dict(arrowstyle='simple', fc='red', lw=1))


# # 3

# ### Pie chart

data_set = np.array([47, 30, 12, 37, 6, 28, 9, 11, 1])
my_labels = ["Toyota", "Audi", "Bentley", "Mini", "Lamborgini", "BMW", "Ferrari", "Alfa Romeo", "Bugatti"]
explode = [0.02]*len(data_set)

plt.pie(data_set, labels = my_labels, explode=explode, )
plt.show()


