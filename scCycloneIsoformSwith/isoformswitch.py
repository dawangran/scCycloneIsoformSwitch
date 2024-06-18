# -*- coding: utf-8 -*-
"""
@File    :   isoformswith.py
@Time    :   2024/05/29 13:39:55
@Author  :   Dawn
@Version :   1.0
@Desc    :   Isoformswith for single cell
"""


import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from functools import reduce
from datetime import datetime
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("split_sample.log"),
        logging.StreamHandler()
    ]
)

def split_sample(adata, cell_label, control, treatment, fraction=0.1, random_state=0):
    """
    Randomly sample an AnnData object based on control and treatment groups to generate a subset.

    Parameters:
        adata (sc.AnnData): AnnData object containing single-cell RNA sequencing data.
        cell_label (str): Label indicating cell types, must be a column in adata.obs.
        control (str): Label value for the control group.
        treatment (str): Label value for the treatment group.
        fraction (float): Sampling fraction, between 0 and 1.
        random_state (int): Random seed used for reproducibility of results.

    Returns:
        sc.AnnData: Generated subset containing randomly sampled data from control and treatment groups.
    """
    
    try:
        logging.info("Split adata start!")
        
        # 验证输入参数
        if not isinstance(adata, sc.AnnData):
            raise ValueError("adata should be an AnnData object")
        if cell_label not in adata.obs.columns:
            raise ValueError(f"cell_label '{cell_label}' not found in adata.obs")
        if control not in adata.obs[cell_label].unique():
            raise ValueError(f"control '{control}' not found in adata.obs['{cell_label}']")
        if treatment not in adata.obs[cell_label].unique():
            raise ValueError(f"treatment '{treatment}' not found in adata.obs['{cell_label}']")
        if not (0 < fraction <= 1):
            raise ValueError("fraction should be between 0 and 1")

        # 筛选控制组和处理组数据
        adata_c = adata[adata.obs[cell_label] == control]
        adata_t = adata[adata.obs[cell_label] == treatment]

        # 随机抽样
        adata_c_s = sc.pp.subsample(adata_c, fraction=fraction, copy=True, random_state=random_state)
        adata_t_s = sc.pp.subsample(adata_t, fraction=fraction, copy=True, random_state=random_state)

        # 合并子数据集
        adata_meta = sc.concat([adata_c_s, adata_t_s])
        adata_meta.var = adata.var

        logging.info("Split adata success!")
        return adata_meta

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        
        
def filter_adata(adata, cell_label, control, treatment, gene_label, percent=0.3):
    """
    Filter an AnnData object to retain genes that are expressed in both control and treatment groups.

    Parameters:
        adata (sc.AnnData): AnnData object containing single-cell RNA sequencing data.
        cell_label (str): Label indicating cell types, must be a column in adata.obs.
        control (str): Label value for the control group.
        treatment (str): Label value for the treatment group.
        gene_label (str): Label indicating genes, must be a column in adata.var.
        percent (float): Percentage threshold for filtering genes, between 0 and 1.

    Returns:
        sc.AnnData: Filtered AnnData object.
    """
    try:
        logging.info("Filter adata start!")
        
        # 验证输入参数
        if not isinstance(adata, sc.AnnData):
            raise ValueError("adata should be an AnnData object")
        if cell_label not in adata.obs.columns:
            raise ValueError(f"cell_label '{cell_label}' not found in adata.obs")
        if control not in adata.obs[cell_label].unique():
            raise ValueError(f"control '{control}' not found in adata.obs['{cell_label}']")
        if treatment not in adata.obs[cell_label].unique():
            raise ValueError(f"treatment '{treatment}' not found in adata.obs['{cell_label}']")
        if gene_label not in adata.var.columns:
            raise ValueError(f"gene_label '{gene_label}' not found in adata.var")
        if not (0 <= percent <= 1):
            raise ValueError("percent should be between 0 and 1")

        # 筛选控制组和处理组数据
        adata_c = adata[adata.obs[cell_label] == control]
        adata_t = adata[adata.obs[cell_label] == treatment]

        if percent==0:
            pass
        else:
            # 过滤基因
            sc.pp.filter_genes(adata_c, min_cells=int(adata_c.shape[0] * percent))
            sc.pp.filter_genes(adata_t, min_cells=int(adata_t.shape[0] * percent))

        # 获取两个条件下共同的基因列表
        gene_list = list(set(adata_c.var[gene_label]) & set(adata_t.var[gene_label]))
        iso_list = adata.var[adata.var[gene_label].isin(gene_list)].index.to_list()

        # 过滤原始数据
        adata = adata[:, iso_list]

        logging.info("Filter adata success!")
        return adata

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def generate_gene_iso(adata, gene_label):
    """
    Generate a dictionary of gene isoforms.

    Parameters:
        adata (sc.AnnData): AnnData object containing gene expression data.
        gene_label (str): Label indicating genes, must be a column in adata.var.

    Returns:
        dict: Dictionary of gene isoforms, where keys are gene names and values are lists of isoforms associated with each gene.
    """
    try:
        logging.info("Step 1: Generate gene iso start!")
        
        # 验证输入参数
        if not isinstance(adata, sc.AnnData):
            raise ValueError("adata should be an AnnData object")
        if gene_label not in adata.var.columns:
            raise ValueError(f"gene_label '{gene_label}' not found in adata.var")

        gene_data = adata.var[[gene_label]]
        
        gene_iso = {}
        for k, v in zip(gene_data[gene_label], gene_data.index):
            if k not in gene_iso:
                gene_iso[k] = [v]
            else:
                gene_iso[k].append(v)

        logging.info("Step 1: Generate gene iso success!")
        return gene_iso

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        
        
        
def process_adata(adata, gene_iso):
    """
    Filter genes of an AnnData object based on a provided dictionary of gene isoforms.

    Parameters:
        adata (sc.AnnData): AnnData object containing gene expression data.
        gene_iso (dict): Dictionary of gene isoforms, where keys are gene names and values are lists of isoforms associated with each gene.

    Returns:
        sc.AnnData: Filtered AnnData object, retaining only the specified gene isoforms.
    """
    try:
        logging.info("Process adata start!")

        # 验证输入参数
        if not isinstance(adata, sc.AnnData):
            raise ValueError("adata should be an AnnData object")
        if not isinstance(gene_iso, dict):
            raise ValueError("gene_iso should be a dictionary")
        for k, v in gene_iso.items():
            if not isinstance(k, str):
                raise ValueError("gene_iso keys should be strings representing gene names")
            if not isinstance(v, list):
                raise ValueError("gene_iso values should be lists of gene isoforms")
            for iso in v:
                if iso not in adata.var.index:
                    raise ValueError(f"isoform '{iso}' not found in adata.var.index")

        # 提取所有基因异构体的列表
        iso_list = [iso for isos in gene_iso.values() for iso in isos]

        # 筛选 AnnData 对象中的基因
        adata = adata[:, iso_list]

        logging.info("Process adata success!")
        return adata

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def generate_IF_matrix(adata, gene_iso, control, treatment, cell_label, gene_label):
    """
    Generate an IF (Isoform Fraction) matrix based on the input AnnData object.

    Parameters:
        adata (sc.AnnData): AnnData object containing single-cell RNA sequencing data.
        gene_iso (dict): Dictionary mapping genes to their corresponding isoforms, where keys are gene names and values are lists of isoforms.
        control (str): Label value for the control group.
        treatment (str): Label value for the treatment group.
        cell_label (str): Label indicating cell types, must be a column in adata.obs.
        gene_label (str): Label indicating gene names, must be a column in adata.var.

    Returns:
        pd.DataFrame: Generated IF matrix.
    """

    try:
        current_time = datetime.now()
        logging.info("Step 3: Generate IF matrix start! {}".format(current_time))
        
        # 验证输入参数
        if not isinstance(adata, sc.AnnData):
            raise ValueError("adata should be an AnnData object")
        if cell_label not in adata.obs.columns:
            raise ValueError(f"cell_label '{cell_label}' not found in adata.obs")
        if gene_label not in adata.var.columns:
            raise ValueError(f"gene_label '{gene_label}' not found in adata.var")
        if control not in adata.obs[cell_label].unique():
            raise ValueError(f"control '{control}' not found in adata.obs['{cell_label}']")
        if treatment not in adata.obs[cell_label].unique():
            raise ValueError(f"treatment '{treatment}' not found in adata.obs['{cell_label}']")
        if not isinstance(gene_iso, dict):
            raise ValueError("gene_iso should be a dictionary")
        
        # 筛选控制组和处理组数据
        adata_sub = adata[adata.obs[cell_label].isin([control, treatment])]
        
        # 转换为 DataFrame
        data = adata_sub.to_df()
        data[cell_label] = adata_sub.obs[cell_label].to_list()
        
        # 按 cell_label 分组并聚合数据
        data = data.groupby(cell_label).agg(sum)
        
        # 构建 IF 矩阵
        data_IF_list = []
        for gene, isoforms in gene_iso.items():
            data_sub = data[isoforms]
            row_sums = data_sub.sum(axis=1)  # 计算每行的和
            data_IF_sub = data_sub.div(row_sums, axis=0)  # 每个元素除以所在行的和
            data_IF_list.append(data_IF_sub)
        
        # 合并各基因的 IF 子矩阵
        data_IF = pd.concat(data_IF_list, axis=1)
        data_IF = data_IF.T
        data_IF = data_IF.fillna(0)
        
        current_time = datetime.now()
        logging.info("Step 3: Generate IF matrix success! {}".format(current_time))
        
        return data_IF

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        
        
        
def generate_gene_rank(data_IF, gene_iso, control, treatment):
    """
    Generate gene ranking based on the IF matrix and find the index of the transcript with the highest expression for each gene.

    Parameters:
        data_IF (pd.DataFrame): IF (Isoform Fraction) matrix.
        gene_iso (dict): Dictionary mapping genes to their corresponding isoforms, where keys are gene names and values are lists of isoforms.
        control (str): Label value for the control group.
        treatment (str): Label value for the treatment group.

    Returns:
        tuple: Contains two dictionaries, one for gene ranking and one for the index of the transcript with the highest expression.
    """
    try:
        current_time = datetime.now()
        logging.info("Step 4: Generate gene rank start! {}".format(current_time))
        
        # 验证输入参数
        if not isinstance(data_IF, pd.DataFrame):
            raise ValueError("data_IF should be a pandas DataFrame")
        if not isinstance(gene_iso, dict):
            raise ValueError("gene_iso should be a dictionary")
        if control not in data_IF.columns:
            raise ValueError(f"control '{control}' not found in data_IF columns")
        if treatment not in data_IF.columns:
            raise ValueError(f"treatment '{treatment}' not found in data_IF columns")
        
        gene_rank = {}
        gene_max_index = {}
        gene_IF = {}
        gene_change_iso={}
        for gene, isoforms in gene_iso.items():
            if not all(isoform in data_IF.index for isoform in isoforms):
                logging.warning(f"Not all isoforms of gene '{gene}' are present in data_IF")
                continue
            
            df = data_IF.loc[isoforms]
            rank_data = df.apply(lambda column: column.rank(ascending=False), axis=0)
            
            control_rank_list = rank_data[control].to_list()
            treatment_rank_list = rank_data[treatment].to_list()

            control_IF_list = df[control].to_list()
            treatment_IF_list = df[treatment].to_list()
            
            max_control_index = control_rank_list.index(min(control_rank_list))
            max_treatment_index = treatment_rank_list.index(min(treatment_rank_list))
            
            different_positions = [
                i for i, (x, y) in enumerate(zip(control_rank_list, treatment_rank_list)) if x != y
            ]
            
            different_value = [
                [control_IF_list[i],treatment_IF_list[i]] for i in  different_positions
            ]

            different_iso = [
                rank_data.index.to_list()[i] for i in  different_positions
            ]
        
            if (len(different_positions) != 0):
                
                gene_rank[gene] = different_positions
                gene_max_index[gene] = [max_control_index,max_treatment_index]
                gene_IF[gene] = different_value
                gene_change_iso[gene] = different_iso
                
        
        current_time = datetime.now()
        logging.info("Step 4: Generate gene rank success! {}".format(current_time))
        
        return gene_rank, gene_max_index,gene_IF,gene_change_iso

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")



def generate_rss(data_IF, gene_rank, gene_iso, control, treatment):
    """
    Generate gene RSS (Residual Sum of Squares) based on the IF matrix and gene ranking.

    Parameters:
        data_IF (pd.DataFrame): IF (Isoform Fraction) matrix.
        gene_rank (dict): Dictionary of gene ranking, where keys are gene names and values are lists of ranking indices.
        gene_iso (dict): Dictionary mapping genes to their corresponding isoforms, where keys are gene names and values are lists of isoforms.
        control (str): Label value for the control group.
        treatment (str): Label value for the treatment group.

    Returns:
        tuple: Contains two dictionaries, one for gene RSS and one for state RSS.
    """
    try:
        current_time = datetime.now()
        logging.info("Step 5: Generate gene RSS start! {}".format(current_time))
        
        # 验证输入参数
        if not isinstance(data_IF, pd.DataFrame):
            raise ValueError("data_IF should be a pandas DataFrame")
        if not isinstance(gene_rank, dict):
            raise ValueError("gene_rank should be a dictionary")
        if not isinstance(gene_iso, dict):
            raise ValueError("gene_iso should be a dictionary")
        if control not in data_IF.columns:
            raise ValueError(f"control '{control}' not found in data_IF columns")
        if treatment not in data_IF.columns:
            raise ValueError(f"treatment '{treatment}' not found in data_IF columns")
        
        gene_rss = {}
        state_rss = {}
        state_rss_value = {}
        
        for gene, indices in gene_rank.items():
            isoforms = gene_iso[gene]
            df = data_IF.loc[isoforms]
            
            residuals = df[treatment] - df[control].to_list()
            residuals_filter = np.array([residuals[i] for i in indices])
            
            state_rss[gene] = ["up" if r > 0 else "down" for r in residuals_filter]
            state_rss_value[gene] = [r for r in residuals_filter]
            residual_sum_of_squares = np.sum((residuals_filter * 10) ** 2) / len(indices)
            gene_rss[gene] = residual_sum_of_squares
        
        current_time = datetime.now()
        logging.info("Step 5: Generate gene RSS success! {}".format(current_time))
        
        return gene_rss, state_rss,state_rss_value

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        
        

def concat_gene_rank_rss(gene_rank, gene_IF, gene_change_iso, gene_iso,gene_rss, state_rss, state_rss_value,gene_max_index):
    """
    Generate gene RSS (Residual Sum of Squares) based on the IF matrix and gene ranking.

    Parameters:
       
        gene_rank (dict): Dictionary of gene ranking, where keys are gene names and values are lists of ranking indices.
        gene_IF (pd.DataFrame): IF (Isoform Fraction) matrix.
        gene_max_index (dict): Dictionary of indices of the transcript with the highest expression for each gene.
        gene_change_iso (dict): Dictionary mapping genes to their corresponding isoforms, where keys are gene names and values are lists of isoforms.
        
    Returns:
        tuple: Contains two dictionaries, one for gene RSS and one for state RSS.
    """
    try:
        current_time = datetime.now()
        logging.info("Step 6: Generate result start! {}".format(current_time))
        
        # 验证输入参数
        if not isinstance(gene_rank, dict):
            raise ValueError("gene_rank should be a dictionary")
        if not isinstance(gene_IF, dict):
            raise ValueError("gene_IF should be a dictionary")
        if not isinstance(gene_change_iso, dict):
            raise ValueError("gene_change_iso should be a dictionary")
        if not isinstance(gene_rss, dict):
            raise ValueError("gene_rss should be a dictionary")
        if not isinstance(state_rss, dict):
            raise ValueError("state_rss should be a dictionary")
        if not isinstance(state_rss_value, dict):
            raise ValueError("state_rss_value should be a dictionary")
        if not isinstance(gene_max_index, dict):
            raise ValueError("gene_max_index should be a dictionary")
        
        # 创建包含基因排序信息的数据框
        gene_rank_data = pd.DataFrame({
            "gene_name": gene_rank.keys(),
            "site": list(gene_rank.values()),
            "max_site": list(gene_max_index.values()),
            "IF": list(gene_IF.values()),
            "isoform": list(gene_change_iso.values()),
        }).set_index("gene_name")
        
        max_isoform_list=[]
        for k,v in zip(gene_rank_data.index,gene_rank_data["max_site"]):
            max_isoform_list.append([gene_iso[k][i] for i in v])
        gene_rank_data['max_isoform']=max_isoform_list
        
        # 创建包含基因RSS和状态信息的数据框
        gene_rss_data = pd.DataFrame({
            "gene_name": gene_rss.keys(),
            "rss": list(gene_rss.values()),
            "state": list(state_rss.values()),
            "rss_info": list(state_rss_value.values())
        }).set_index("gene_name")
        
        # 合并数据框
        data = pd.concat([gene_rank_data, gene_rss_data], axis=1)
        
        # 计算每个基因的站点数量
        data['number'] = [len(sites) for sites in data['site']]
        
        current_time = datetime.now()
        logging.info("Step 6: Generate result success! {}".format(current_time))
        
        return data

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        

def add_Mann_Whitney(adata, data, control, treatment, cell_label):
    """
    Add Mann-Whitney U test results to the DataFrame.

    Parameters:
    adata : AnnData
        Annotated data matrix.
    data : DataFrame
        DataFrame containing the isoform switch analysis results.
    control : str
        Label of the control condition.
    treatment : str
        Label of the treatment condition.
    cell_label : str
        Label containing cell type information.

    Returns:
    data : DataFrame
        DataFrame with added columns for p-values and their significance states.
    """

    # Input validation
    if not isinstance(adata, sc.AnnData):
        logging.error("Invalid data type for adata. Expected AnnData.")
        raise ValueError("Invalid data type for adata. Expected AnnData.")
    
    if not isinstance(data, pd.DataFrame):
        logging.error("Invalid data type for data. Expected DataFrame.")
        raise ValueError("Invalid data type for data. Expected DataFrame.")
    
    if control not in adata.obs[cell_label].unique():
        logging.error(f"Control '{control}' not found in '{cell_label}' labels of adata.")
        raise ValueError(f"Control '{control}' not found in '{cell_label}' labels of adata.")
    
    if treatment not in adata.obs[cell_label].unique():
        logging.error(f"Treatment '{treatment}' not found in '{cell_label}' labels of adata.")
        raise ValueError(f"Treatment '{treatment}' not found in '{cell_label}' labels of adata.")
    
    # Perform Mann-Whitney U test for each gene
    gene_pval_list = []

    data_IF_control = adata[adata.obs[cell_label]==control].to_df()
    data_IF_treatment = adata[adata.obs[cell_label]==treatment].to_df()

    for i in data['isoform']:
        pval_list=[]
        for j in i:
            group1 =data_IF_control[j].to_list()
            group2 =data_IF_treatment[j].to_list()
            u, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            pval_list.append(p_value)
        gene_pval_list.append(pval_list)

    data['pval'] = gene_pval_list
    
    return data
    
        
def isoformwithAnalysis(adata_count, adata_IF, control, treatment, cell_label, gene_label):
    """
    Perform Isoform Switch analysis.

    Parameters:
        adata (AnnData): Input AnnData object.
        adata_IF (AnnData): Input AnnData object.
        control (str): Name of the control group.
        treatment (str): Name of the treatment group.
        cell_label (str): Column name for cell labels.
        gene_label (str): Column name for gene labels.

    Returns:
        tuple: A tuple containing the following elements:
            data_rs (pd.DataFrame): Result DataFrame containing gene ranking, RSS, and state information.
            data_IF (pd.DataFrame): IF matrix DataFrame.
            gene_iso (pd.DataFrame): Gene isoform DataFrame.
    """
    try:
        current_time = datetime.now()
        logging.info("-------------Isoformswitch Analysis Start--------------")
        logging.info("Isoformswitch analysis start! {}".format(current_time))

        # 验证输入参数
        if not isinstance(control, str):
            raise ValueError("control should be a string")
        if not isinstance(treatment, str):
            raise ValueError("treatment should be a string")
        if not isinstance(cell_label, str):
            raise ValueError("cell_label should be a string")
        if not isinstance(gene_label, str):
            raise ValueError("gene_label should be a string")

        # 生成基因同工型数据
        gene_iso = generate_gene_iso(adata_count, gene_label=gene_label)
        
        # 处理 AnnData 对象
        adata = process_adata(adata_count, gene_iso)
        
        # 生成 IF 矩阵
        data_IF = generate_IF_matrix(adata, gene_iso=gene_iso, control=control, treatment=treatment, cell_label=cell_label, gene_label=gene_label)
        
        # 生成基因排序数据
        gene_rank, gene_max_index,gene_IF,gene_change_iso = generate_gene_rank(data_IF, gene_iso, control=control, treatment=treatment)
        
        # 生成 RSS 数据
        gene_rss, state_rss ,state_rss_value= generate_rss(data_IF, gene_rank=gene_rank, gene_iso=gene_iso, control=control, treatment=treatment)
        
        # 合并基因排序和 RSS 数据
        data_rs = concat_gene_rank_rss(gene_rank=gene_rank, gene_IF=gene_IF, gene_change_iso=gene_change_iso,gene_iso=gene_iso, gene_rss=gene_rss, gene_max_index=gene_max_index, state_rss=state_rss,state_rss_value=state_rss_value)

        # add Mann Whitney
        data_rs = add_Mann_Whitney(adata=adata_IF, data=data_rs, control=control, treatment=treatment, cell_label=cell_label)
        
        
        # 添加条件和批次信息
        data_rs['condition'] = "{}_{}".format(control, treatment)
        data_rs['batch'] = 0
        data_IF['batch'] = 0
        
        current_time = datetime.now()
        logging.info("Isoformswitch analysis success! {}".format(current_time))
        logging.info("-------------Isoformswitch Analysis Success--------------")

        return data_rs, data_IF, gene_iso

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
         
         
def scIsoformwithAnalysis(adata_count, adata_IF, control, treatment,cell_label, gene_label, percent=0.3):
    """
    Perform isoform switch analysis on single-cell RNA-seq data.

    Parameters:
    adata_count : AnnData
        Annotated data matrix.
    adata_IF : AnnData
        Annotated data matrix.
    control : str
        Label of the control condition.
    treatment : str
        Label of the treatment condition.
    cell_label : str
        The label in adata.obs that contains cell type information.
    gene_label : str
        The label in adata.var that contains gene information.
    percent : float, optional (default=0.3)
        The percentage threshold for filtering genes.

    Returns:
    data_rs : DataFrame
        Results of the isoform switch analysis.
    data_IF : DataFrame
        Isoform fractions data.
    gene_iso : DataFrame
        Gene isoform information.
    """

    current_time = datetime.now()
    logging.info("-------------Isoformswitch Analysis Start--------------")
    logging.info("Isoformswitch analysis start! {}".format(current_time))
    
    # Input validation
    if not isinstance(adata_count, sc.AnnData):
        logging.error("Invalid data type for adata. Expected AnnData.")
        raise ValueError("Invalid data type for adata. Expected AnnData.")

    if not isinstance(adata_IF, sc.AnnData):
        logging.error("Invalid data type for adata. Expected AnnData.")
        raise ValueError("Invalid data type for adata. Expected AnnData.")
    
    if not isinstance(control, str) or not isinstance(treatment, str):
        logging.error("Control and treatment labels must be strings.")
        raise ValueError("Control and treatment labels must be strings.")
    
    if not isinstance(cell_label, str) or not isinstance(gene_label, str):
        logging.error("Cell and gene labels must be strings.")
        raise ValueError("Cell and gene labels must be strings.")
    
    if not (0 <= percent <= 1):
        logging.error("Percent must be between 0 and 1.")
        raise ValueError("Percent must be between 0 and 1.")
    
    try:
        # Filtering the data based on control, treatment, cell label, gene label, and percent
        adata = filter_adata(adata_count, control=control, treatment=treatment, cell_label=cell_label, gene_label=gene_label, percent=percent)
        
        # Performing isoform switch analysis
        data_rs, data_IF, gene_iso = isoformwithAnalysis(adata, adata_IF=adata_IF, control=control, treatment=treatment, cell_label=cell_label, gene_label=gene_label)
        
        current_time = datetime.now()
        logging.info("Isoformswitch analysis success! {}".format(current_time))
        logging.info("-------------Isoformswitch Analysis Success--------------")
        
        return data_rs, data_IF, gene_iso
    
    except Exception as e:
        logging.error("Error during isoform switch analysis: {}".format(e))
        raise



def switchFilter(data, rss=0.2, number=None, first=False, pval=0.05):
    """
    Filter data based on specified thresholds and conditions for isoform switch analysis.

    Parameters:
    data : DataFrame
        Input data containing isoform switch analysis results.
    thr : float, optional (default=0.2)
        Threshold for the rss value. The threshold is squared and multiplied by 100.
    number : int, optional
        Maximum number of isoform switches allowed.
    first : bool, optional (default=False)
        If True, only consider entries where max_site is within the site list.
    pval : float, optional (default=0.05)
        P-value threshold for significance in the Mann-Whitney U test.

    Returns:
    data_filter : DataFrame
        Filtered data.
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        logging.error("Invalid data type for data. Expected DataFrame.")
        raise ValueError("Invalid data type for data. Expected DataFrame.")
    
    if not (0 <= rss <= 1):
        logging.error("Threshold thr must be between 0 and 1.")
        raise ValueError("Threshold thr must be between 0 and 1.")

    if not (0 <= pval <= 1):
        logging.error("Threshold thr must be between 0 and 1.")
        raise ValueError("Threshold thr must be between 0 and 1.")
    
    if number is not None and not isinstance(number, int):
        logging.error("Number must be an integer.")
        raise ValueError("Number must be an integer.")
    
    if not isinstance(first, bool):
        logging.error("First must be a boolean value.")
        raise ValueError("First must be a boolean value.")
    
    
    # Filter based on rss threshold
    rss_thr = (rss * 10) ** 2
    data_filter = data[data["rss"] >= rss_thr]
    logging.info("Filtered data based on rss threshold.")
    
    # Filter based on state containing exactly two unique elements
    data_filter = data_filter[[len(set(i)) == 2 for i in data_filter['state']]]
    logging.info("Filtered data based on state condition.")
    
    # Further filtering if number parameter is provided
    if number and number >= 2:
        data_filter = data_filter[data_filter['number'] <= number]
        logging.info("Filtered data based on number condition.")
    
    # Further filtering if first parameter is True
    if first:
        max_index = [True if len(set(i) & set(j))>0 else False for i, j in zip(data_filter['site'], data_filter['max_site'])]
        data_filter = data_filter[max_index]
        logging.info("Filtered data based on first condition.")
        
    pval_state_list=[]
    if pval:
        for i in data_filter['pval']:
            if sum(1 for x in  i if x < pval) >=1:
                pval_state_list.append(True)
            else:
                pval_state_list.append(False)
    data_filter = data_filter[pval_state_list]   
    
    data_filter = data_filter.sort_values("rss", ascending=False)
    data_filter['batch_num'] = 1
    logging.info("Sorted data by rss in descending order.")
    
    return data_filter


def scIsoformswitchAnalysisBatch(adata_count, adata_IF,control, treatment, cell_label, gene_label, percent=0.3, fraction=0.3, batch=10):
    """
    Perform batch Isoform Switch analysis.

    Parameters:
        adata_count (AnnData): Input AnnData object.
        adata_IF (AnnData): Input AnnData object.
        control (str): Name of the control group.
        treatment (str): Name of the treatment group.
        cell_label (str): Column name for cell labels.
        gene_label (str): Column name for gene labels.
        percent (float): Percentage threshold for filtering genes, between 0 and 1.
        fraction (float): Fraction of data to sample, between 0 and 1.
        batch (int): Number of batches to perform.

    Returns:
        tuple: A tuple containing the following elements:
            data_rs (pd.DataFrame): Result DataFrame containing gene ranking, RSS, and state information.
            data_IF (pd.DataFrame): IF matrix DataFrame.
            gene_iso (pd.DataFrame): Gene isoform DataFrame.
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    data_rs_list = []
    data_IF_list = []
    
    for i in range(batch):
        logging.info("==============Isoformswitch Analysis Start Batch {} ==============".format(i))
        current_time = datetime.now()
        logging.info("Run start batch{} : {}".format(i, current_time))
        
        # Filter the AnnData object
        adata_filtered = filter_adata(adata_count, control=control, treatment=treatment, cell_label=cell_label, gene_label=gene_label, percent=percent)
        
        # Sample a subset of data
        adata_sub = split_sample(adata_filtered, cell_label=cell_label, control=control, treatment=treatment, fraction=fraction, random_state=i)
        
        # Perform Isoform Switch analysis on the subset
        data_rs_sub, data_IF_sub, gene_iso = isoformwithAnalysis(adata_sub, adata_IF=adata_IF, control=control, treatment=treatment, cell_label=cell_label, gene_label=gene_label)
        
        # Add batch information to the result DataFrames
        data_rs_sub['batch'] = i
        data_IF_sub['batch'] = i
        
        # Append the results to the lists
        data_rs_list.append(data_rs_sub)
        data_IF_list.append(data_IF_sub)
        
        current_time = datetime.now()
        logging.info("Success batch{} : {}".format(i, current_time))  
        logging.info("==============Isoformswitch Analysis Success Batch {} ==============".format(i))
    
    # Concatenate the results from all batches
    data_rs = pd.concat(data_rs_list)
    data_IF = pd.concat(data_IF_list)
    
    return data_rs, data_IF, gene_iso



# def scIsoformswitchAnalysisBatch(adata_count, adata_IF,control, treatment, cell_label, gene_label, percent=0.3, fraction=0.3, batch=10):
#     """
#     Perform batch Isoform Switch analysis.

#     Parameters:
#         adata_count (AnnData): Input AnnData object.
#         adata_IF (AnnData): Input AnnData object.
#         control (str): Name of the control group.
#         treatment (str): Name of the treatment group.
#         cell_label (str): Column name for cell labels.
#         gene_label (str): Column name for gene labels.
#         percent (float): Percentage threshold for filtering genes, between 0 and 1.
#         fraction (float): Fraction of data to sample, between 0 and 1.
#         batch (int): Number of batches to perform.

#     Returns:
#         tuple: A tuple containing the following elements:
#             data_rs (pd.DataFrame): Result DataFrame containing gene ranking, RSS, and state information.
#             data_IF (pd.DataFrame): IF matrix DataFrame.
#             gene_iso (pd.DataFrame): Gene isoform DataFrame.
#     """
#     # Initialize logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
#     data_rs_list = []
#     data_IF_list = []
    
#     for i in range(batch):
#         logging.info("==============Isoformswitch Analysis Start Batch {} ==============".format(i))
#         current_time = datetime.now()
#         logging.info("Run start batch{} : {}".format(i, current_time))
        
#         # Filter the AnnData object
#         adata_filtered = filter_adata(adata_count, control=control, treatment=treatment, cell_label=cell_label, gene_label=gene_label, percent=percent)
        
#         # Sample a subset of data
#         adata_sub = split_sample(adata_filtered, cell_label=cell_label, control=control, treatment=treatment, fraction=fraction, random_state=i)
        
#         # Perform Isoform Switch analysis on the subset
#         data_rs_sub, data_IF_sub, gene_iso = isoformwithAnalysis(adata_sub, adata_IF=adata_IF, control=control, treatment=treatment, cell_label=cell_label, gene_label=gene_label)
        
#         # Add batch information to the result DataFrames
#         data_rs_sub['batch'] = i
#         data_IF_sub['batch'] = i
        
#         # Append the results to the lists
#         data_rs_list.append(data_rs_sub)
#         data_IF_list.append(data_IF_sub)
        
#         current_time = datetime.now()
#         logging.info("Success batch{} : {}".format(i, current_time))  
#         logging.info("==============Isoformswitch Analysis Success Batch {} ==============".format(i))
    
#     # Concatenate the results from all batches
#     data_rs = pd.concat(data_rs_list)
#     data_IF = pd.concat(data_IF_list)
    
#     return data_rs, data_IF, gene_iso


# def swicthFilter(data, rss=0.2, number=None, first=False, accuracy=1, force=True, pval=0.05):
#     """
#     Filter the data based on RSS (Residual Sum of Squares), number of isoforms, and p-value.

#     Parameters:
#         data (pd.DataFrame): DataFrame containing the data to be filtered.
#         thr (float): Threshold for RSS.
#         number (int): Maximum number of isoforms allowed.
#         first (bool): Whether to keep only the isoform with the highest expression.
#         accuracy (float): Accuracy of batch selection.
#         force (bool): Whether to force filtering based on isoform intersection.
#         pval (float): p-value threshold for filtering.

#     Returns:
#         pd.DataFrame: Filtered DataFrame.
#     """
#     rss_thr = (rss * 10) ** 2
#     data_filter = data[data["rss"] >= rss_thr]
#     data_filter = data_filter.loc[[True if len(set(i)) == 2 else False for i in data_filter['state']]]
    
#     if number and number >= 2:
#         data_filter = data_filter[data_filter['number'] <= number]
    
#     if first:
#         max_index = [j in i for i, j in zip(data_filter['site'], data_filter['max_site'])]
#         data_filter = data_filter[max_index]
    
#     data_filter = data_filter.sort_values("rss", ascending=False)
    
#     pval_state_list = []
#     if pval:
#         for i in data_filter['pval']:
#             if sum(1 for x in i if x < pval) >= 1:
#                 pval_state_list.append(True)
#             else:
#                 pval_state_list.append(False)
#     data_filter = data_filter[pval_state_list] 

#     data_filter['batch_num'] = 1
#     data_filter_batch_number = len(set(data_filter['batch']))
#     batch_num = int(data_filter_batch_number * accuracy)
    
#     if 1 < batch_num <= data_filter_batch_number:
#         data_filter_meta = data_filter.groupby("gene_name").agg(list)
#         data_filter_meta['batch_num'] = [len(i) for i in data_filter_meta['batch']]
#         data_filter_meta['condition'] = [i[0] for i in data_filter_meta['condition']]
#         data_filter_meta['max_site'] = [i[0] for i in data_filter_meta['max_site']]
        
#         data_filter_meta = data_filter_meta[data_filter_meta['batch_num'] > 1]
#         data_filter_meta['rss_mean'] = [sum(i) / len(i) for i in data_filter_meta['rss']]
#         data_filter = data_filter_meta[data_filter_meta['batch_num'] >= batch_num]
        
#         tag_list = []
#         for i in data_filter['site']:
#             intersection = reduce(lambda x, y: set(x) & set(y), i)
#             min_length = min(len(sublist) for sublist in i)
#             if len(intersection) >= min_length:
#                 tag_list.append(True)
#             else:
#                 tag_list.append(False)
        
#         data_filter['Tag'] = tag_list 
        
#         if force:
#             data_filter = data_filter[data_filter['Tag'] == True]
    
#     return data_filter
