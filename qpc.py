# -*- coding: utf-8 -*-
"""
# Author: Li Xiang@CICC.EQ
# Created Time : Thu 08 Aug 2019 03:31:09 PM CST
# File Name: qpc.py
# Description:
"""

#%% Import Part
import os,sys,re
import numpy as np
import pandas as pd
import cx_Oracle as ora
import pymysql
import datetime as dm
from qp import *
import warnings
warnings.filterwarnings("ignore")
import pdb

#%% Global Variables
DATAPATH = '/eq/share/lix/data'

#%% Class of MemoryMap
class mmap():
    o = dict()
    e = dict()
    v = dict()

#%% Class of GlobalVars
class gvars():
    # Trade Dates
    StartDate = '20060101'
    TradeDateFile = os.path.join(DATAPATH,'tmp/trade.date.txt')
    # Ids
    IdFile = os.path.join(DATAPATH,'tmp/ids.txt')
    # Base ini
    BaseIni = os.path.join(DATAPATH,'ini/cn.eq.base.ini')
    # Tick data
    TickDataPath = os.path.join(DATAPATH,'tmp/rq/raw/tick')
    # MB1 data
    MB1DataPath = os.path.join(DATAPATH,'tmp/rq/csv/mb1')
    # Mysql connection
    WinddfInfo = {'host':'localhost','db':'WINDDF','user':'readonly',\
        'passwd':'read123','charset':'utf8'}
    # Id components
    IdComponets = ['AShareStocks','AShareIndices']

#%% Class of base
class base_():
    def __init__(self,fini=None):
        if fini is not None:
            self.fini = fini
            self.ini = Ini(fini)

#%% Class of Instruments
class Instruments():
    pass

class AShareIndices(Instruments):
    tickers = ['csi300',
               'csi500',
               'sse50',
               'csi800'] 
    tickers_sh = ['000300',
                  '000905',
                  '000016',
                  '000906']

class AShareStocks(Instruments):
    @property
    def tickers(self):
        sql = '''
              select S_INFO_CODE from winddf.AShareDescription 
              where S_INFO_LISTDATE is not null
              order by S_INFO_CODE
              '''.upper()
        conn = pymysql.connect(**gvars.WinddfInfo)
        ll = pd.read_sql(sql,conn)['S_INFO_CODE'].tolist()
        conn.close()
        return ll

#%% Functions
def list_index(list_arr:list,index_arr:list)->list:
    return [list_arr[ii] for ii in index_arr]

def index_lshort_in_llong(lshort:list,llong:list)->np.ndarray:
    return np.array([llong.index(ss) for ss in lshort])

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def yyyymmdd2yyyy_mm_dd(yyyymmdd):
    return '-'.join([yyyymmdd[:4],yyyymmdd[4:6],yyyymmdd[-2:]])

#---------------------- DataDictionary ----------------------
def generate_dictionary_item(type:str='UniverseBinary',path:str='',configs:str='mmap')->str:
    '''
    type:   'UniverseBinary'/'UniverseBinary3D'/'AMDB'
    '''
    item = ':'.join([type,path,configs])
    return item

def parse_dictionary_item(item:str)->dict:
    type,path,*configs = item.split(':')
    return {'type':type,'path':path,'configs':configs}

def generate_dictionary_ini_file(inifile:str,dirpath:str,datatype:str='UniverseBinary',configs:str='mmap',mode:str='w'):
    '''
    mode: 
        'w', rewrite the inifile.
        'a', subsequent write to the inifile
    '''
    file_list = os.listdir(dirpath)
    file_list = [f for f in file_list if f[-4:]=='.bin']
    file_list.sort()
    with open(inifile,mode) as f:
        if mode=='a':f.write('\n')
        f.write('[DataDictionary] # Created at {}\n'.format(dm.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        for file in file_list:
            path = os.path.abspath(os.path.join(dirpath,file))
            item = generate_dictionary_item(datatype,path,configs)
            f.write('{0} = [ {1} ]\n'.format(file[:-4],item))

#---------------------- Date ----------------------
def generate_trade_date_file():
    sql = "SELECT \
           TRADE_DAYS \
           FROM \
           WINDDF.ASHARECALENDAR \
           WHERE \
           S_INFO_EXCHMARKET = 'SSE' AND \
           TRADE_DAYS >= {} \
           ORDER BY \
           TRADE_DAYS ASC".format('10000000')
    conn = pymysql.connect(**gvars.WinddfInfo)
    dts = pd.read_sql(sql,conn)
    dts.to_csv(gvars.TradeDateFile,header = False,index = False)
    conn.close()

def all_trade_dates()->list:
    try:
        with open(gvars.TradeDateFile,'r') as f:dates = f.read().splitlines()
    except:
        dates = []
    return dates

def n_all_trade_dates()->int:
    return len(open(gvars.TradeDateFile,'r').readlines())

def global_trade_dates()->list:
    return get_dates(sdate=gvars.StartDate,edate=Ini().find('today'))

def datestr2num(datestr:list)->np.ndarray:
    return np.array([int(dt) for dt in datestr])

def datenum2str(datenum:np.ndarray)->list:
    return [str(int(dt)) for dt in datenum]

def date_offset(date:str,offset:int=-1)->str:
    try:
        dates = datestr2num(all_trade_dates())
        dt = dates[np.where(dates>=int(date))[0][0]+offset]
        return str(dt)
    except: return ''

def get_dates(sdate:str=None,edate:str=None,window:int=None,
              dates:list=None,type:str='[]'):
    if dates is None:
        if (sdate is not None) and (edate is not None):
            dates = datestr2num(all_trade_dates())
            dates = dates[(dates>=int(sdate)) & (dates<=int(edate))]
            dates = datenum2str(dates)
        elif (sdate is None) and (edate is not None):
            sdate = date_offset(edate,offset=-window+1)
            dates = get_dates(sdate=sdate,edate=edate)
        elif (edate is None) and (sdate is not None):
            edate = date_offset(sdate,offset=window-1)
            dates = get_dates(sdate=sdate,edate=edate)
    if '(' in type: del dates[0]
    if ')' in type: del dates[-1]
    return dates

def today():
    return dm.datetime.today().strftime('%Y%m%d')

#---------------------- Ids ----------------------
def generate_ids_file():
    ids = []
    for cp in gvars.IdComponets:
        ids += (globals()[cp]().tickers)
    ids.sort()
    with open(gvars.IdFile, "w", newline="") as f:
        for id in ids:f.write("%s\n" % id)

def all_ids()->list:
    try:
        with open(gvars.IdFile,'r') as f:ids = f.read().splitlines()
    except:
        ids = []
    return ids

def n_all_ids()->int:
    return len(open(gvars.IdFile,'r').readlines())

def ids_market(ids:list=None,sh='sh',sz='sz',idx='idx')->list:
    if ids is None:ids = all_ids()
    d = {'00':sz,
         '30':sz,
         '60':sh,
         '68':sh,
         'T0':sh,
         'cs':idx,
         'ss':idx}
    return [d[id[:2]] for id in ids]

def all_ashare_stock_ids()->list:
    ids_db = AShareStocks().tickers
    ids_all = all_ids()
    ids = [id for id in ids_db if id in ids_all]
    return ids

def all_ashare_index_ids()->list:
    ids_db = AShareIndices().tickers
    ids_all = all_ids()
    ids = [id for id in ids_db if id in ids_all]
    return ids

def all_ashare_index_ids_sh()->dict:
    ids_db = AShareIndices().tickers
    ids_db_sh = AShareIndices().tickers_sh
    ids_all = all_ids()
    ids = {id:id_sh for id,id_sh in zip(ids_db,ids_db_sh) if id in ids_all}
    return ids

#---------------------- Read ----------------------
def mreadm2df(file:str):
    '''
    During a running time of a program, contents of a file will be memorized in mmap.o for later use.
    '''
    if file in mmap.o:
        bn = mmap.o[file]
    else:
        if os.path.exists(file):
            bn = readm2df(file)
        else:
            bn = pd.DataFrame()
        mmap.o[file] = bn
    return bn

def mreadm2env(file:str):
    '''
    This function is used to read and memorize ENV-shaped DataFrame of a file.
    During a running time of a program, contents of a file will be memorized in mmap.e for later use.
    The structure stored in mmap.e is a pd.DataFrame.
    '''
    if file in mmap.e:
        df = mmap.e[file]
    else:
        if os.path.exists(file):
            df = readm2df(file)
            df = pd.DataFrame(df,index=global_trade_dates(),columns=all_ids())
        else:
            df = pd.DataFrame(index=global_trade_dates(),columns=all_ids())
        mmap.e[file] = df
    return df

def _merge_(df1,df2):
    # dates
    dts1 = set(df1.index)
    dts2 = set(df2.index)
    dts_merge = list(set.union(dts1,dts2))
    dts_merge.sort()
    dts_new = list(dts2-dts1)
    # tickers
    df = pd.DataFrame(df1,index=dts_merge)
    df.loc[dts_new,:] = df2.loc[dts_new,:]
    return df

def readm2df_from_dictionary(name:str,fini:str=None,fillna=None)->pd.DataFrame:
    if fini is None: fini = gvars.BaseIni
    ini = Ini(fini)
    try:
        svec = ini.findStringVec('DataDictionary~'+name)
    except:
        raise Exception('[{0}] is not in [{1}]!'.format(name,ini.fini))
    data = mreadm2df(parse_dictionary_item(svec[0])['path'])
    if len(svec)>1:
        for ss in svec[1:]:
            data_n = mreadm2df(parse_dictionary_item(ss)['path'])
            data = _merge_(data,data_n)
    if fillna is not None:data.fillna(fillna,inplace=True)
    return data

def readm2env_from_dictionary(name:str,fini:str=None,fillna=None)->pd.DataFrame:
    if fini is None: fini = gvars.BaseIni
    try:
        svec = ini.findStringVec('DataDictionary~'+name)
    except:
        raise Exception('[{0}] is not in [{1}]!'.format(name,ini.fini))
    data = mreadm2env(parse_dictionary_item(svec[0])['path'])
    if len(svec)>1:
        for ss in svec[1:]:
            data_n = mreadm2env(parse_dictionary_item(ss)['path'])
            data = _merge_(data,data_n)
    if fillna is not None:data.fillna(fillna,inplace=True)
    return data

#---------------------- Tick Data ----------------------
def read_tick_data_file(file:str):
    data= pd.read_csv(file,compression='gzip',error_bad_lines=False,index_col=0).dropna()
    data.index = data.index.astype(int)
    data['trading_date'] = data['trading_date'].astype(int)
    return data

def read_tick_data(date:str,ticker:str,fields:list=None):
    '''
    fields: 'trading_date', 'open', 'last', 'high', 'low', 'prev_close', 'volume',
            'total_turnover', 'limit_up', 'limit_down', 'a1', 'a2', 'a3', 'a4',
            'a5', 'b1', 'b2', 'b3', 'b4', 'b5', 'a1_v', 'a2_v', 'a3_v', 'a4_v',
            'a5_v', 'b1_v', 'b2_v', 'b3_v', 'b4_v', 'b5_v', 'change_rate'
    '''
    tick_file = os.path.join(gvars.TickDataPath,date,ticker+'.tar.gz') 
    data = read_tick_data_file(tick_file)
    if fields is not None:
        data = data[fields]
    return data

def read_mb1_data_file(file:str):
    data = pd.read_csv(file,compression='gzip',error_bad_lines=False,dtype={'ticker':str}).dropna()
    data.rename(columns={file.replace('tar.gz','csv'):'time'},inplace=True)
    data['time'] = data['time'].astype(int)
    df = data.set_index(['time','ticker']).unstack()
    df.columns=[t for f,t in df.columns]
    return df

def read_mb1_data(date:str,field:str):
    '''
    ap1  ap3  ap5  av2  av4  bp1  bp3  bp5  bv2  bv4  high      ldvwapsum  limitup  lsp   luvolume   mid   sp  volume  vwapsum
    ap2  ap4  av1  av3  av5  bp2  bp4  bv1  bv3  bv5  ldvolume  limitdown  low      lspp  luvwapsum  open  tp  vwap
    '''
    mb1_file = os.path.join(gvars.MB1DataPath,field,date+'.tar.gz') 
    data = read_mb1_data_file(mb1_file)
    return data
    
#%%
if __name__=='__main__':
    print(today())
    print(date_offset(today(),-20))
