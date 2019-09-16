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
DATAPATH = '/qp/data'

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
    IdDir = os.path.join(DATAPATH,'tmp/ids/')
    # Dir for rq
    RqDir = os.path.join(DATAPATH,'tmp/rq')
    RqIdMapFile = os.path.join(IdDir,'rq/stock.map.csv')
    # Base ini
    BaseIni = os.path.join(DATAPATH,'ini/cn.eq.base.ini')
    # Tick data
    TickDataPath = os.path.join(DATAPATH,'tmp/rq/raw/tick')
    # MB1 data
    MB1DataPath = os.path.join(DATAPATH,'tmp/rq/csv/ashare/mb1')
    # Mysql connection
    WinddfInfo = {'host':'localhost','db':'WINDDF','user':'readonly',\
        'passwd':'read123','charset':'utf8'}
    # Id components
    IdComponets = ['stock','index']
    # Extra tickers
    #ExtraTickers = ['000022','601313']
    ExtraTickers = []

#%% Class of base
class base_():
    def __init__(self,fini=None):
        if fini is not None:
            self.fini = fini
            self.ini = Ini(fini)

#%% Class of Instruments
class Instruments():
    pass

class index(Instruments):
    rq_type = 'INDX'

    tickers = ['csi300',
               'csi500',
               'sse50',
               'csi800', 
               'csi1000'] 
    tickers_sh = ['000300',
                  '000905',
                  '000016',
                  '000906',
                  '000852']

    def tickers_rq(self,dt):
        return pd.Series({'csi300'  :'000300.XSHG',
                          'csi500'  :'000905.XSHG',
                          'sse50'   :'000016.XSHG',
                          'csi800'  :'000906.XSHG',
                          'csi1000' :'000852.XSHG'})

class stock(Instruments):
    rq_type = 'CS'

    @property
    def tickers(self):
        sql = '''
              select S_INFO_CODE from winddf.AShareDescription 
              where S_INFO_LISTDATE is not null
              order by S_INFO_CODE
              '''.upper()
        conn = pymysql.connect(**gvars.WinddfInfo)
        tickers = pd.read_sql(sql,conn)['S_INFO_CODE'].tolist()+gvars.ExtraTickers
        conn.close()
        tickers.sort()
        return tickers

    def tickers_rq(self,dt):
        # sh:'XSHG',sz:'XSHE'
        tkrs_pd = all_ids_types_pd()
        tkrs = tkrs_pd[tkrs_pd=='stock'].index.tolist()
        sh,sz = 'XSHG','XSHE'
        d = {'00':sz,
             '30':sz,
             '60':sh,
             '68':sh,
             'T0':sh}
        tkrs = pd.Series({t:t+'.'+d[t[:2]] for t in tkrs})
        # Read gvars.RqIdMapFile
        dt = int(dt)
        mapping = pd.read_csv(gvars.RqIdMapFile,index_col=0)
        mapping = mapping[(mapping['sdate']<=dt) & (mapping['edate']>=dt)].iloc[:,0]
        tkrs.update(mapping)
        return tkrs

class etf(Instruments):
    rq_type = 'ETF'

class future(Instruments):
    rq_type = 'Future'

class option(Instruments):
    rq_type = 'Option'

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

def yesterday():
    return date_offset(today())

#---------------------- Ids ----------------------
def generate_ids_file():
    old_ids = all_ids()
    ids = []
    types = []
    for cp in gvars.IdComponets:
        tickers = globals()[cp]().tickers 
        stypes = [cp] * len(tickers)
        ids += tickers
        types += stypes
    if set(ids)<set(old_ids):
        print(str(set(old_ids) - set(ids)) + "is not in the new ids!")
        ids = list(set(ids)|set(old_ids))
    with open(gvars.IdFile, "w", newline="") as f:
        for id,type in zip(ids,types):f.write("%s,%s\n" % (id,type))

def all_ids_types()->dict:
    try:
        with open(gvars.IdFile,'r') as f:ids = f.read().splitlines()
        ids_dict = {id.split(',')[0]:id.split(',')[1] for id in ids}
    except:
        ids_dict = {}
    return ids_dict

def all_ids_types_pd(type:'series/df'='series'):
    ids = pd.Series(all_ids_types(),name='type')
    if type=='df': ids = ids.to_frame()
    return ids

def all_ids()->list:
    try:
        with open(gvars.IdFile,'r') as f:ids = f.read().splitlines()
        ids = [id.split(',')[0] for id in ids]
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
    ids_db = stock().tickers
    ids_all = all_ids()
    ids = [id for id in ids_db if id in ids_all]
    return ids

def all_ashare_index_ids()->list:
    ids_db = index().tickers
    ids_all = all_ids()
    ids = [id for id in ids_db if id in ids_all]
    return ids

def all_ashare_index_ids_sh()->dict:
    ids_db = index().tickers
    ids_db_sh = index().tickers_sh
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

# Read 3D-bin
class DataFrame3D():
    def __init__(self,values:'float/int/np.ndarray/DataFrame3D'=np.nan,index:list=None,\
            columns:list=None,depths:list=None):
        if isinstance(values,np.ndarray):
            self.values = values
            self.index = index
            self.columns = columns
            self.depths = depths
            self._check_dims_()
        elif isinstance(values,float) or isinstance(values,int):
            self.index = index
            self.columns = columns
            self.depths = depths
            self.values = np.full([len(index),len(columns),len(depths)],values)
        elif isinstance(values,DataFrame3D):
            if index is None:
                index = values.index
                index_union = index
            else:
                index_union = list(set(index).intersection(set(values.index)))
            if columns is None:
                columns = values.columns
                columns_union = columns
            else:
                columns_union = list(set(columns).intersection(set(values.columns)))
            if depths is None:
                depths = values.depths
                depths_union = depths
            else:
                depths_union = list(set(depths).intersection(set(values.depths)))
            df3 = DataFrame3D(np.full([len(index),len(columns),len(depths)],np.nan),\
                    index,columns,depths)
            df3[index_union,columns_union,depths_union] = values[index_union,columns_union,\
                    depths_union].values
            self.values,self.index,self.columns,self.depths = \
                    df3.values,df3.index,df3.columns,df3.depths

    def _check_dims_(self):
        ii,cc,dd = self.values.shape
        II,CC,DD = self.shape
        if (ii!=II) or (cc!=CC) or (dd!=DD):
            raise ValueError('Shape of passed values is ({}, {}, {}), indices imply ({}, {}, {})'.format(ii,cc,dd,II,CC,DD))

    def __str__(self):
        string = 'Dimensions:   {} (index) x {} (columns) x {} (depths)\nIndex axis:   {} to {}\nColumns axis: {} to {}\nDepths axis:  {} to {}'.format(\
                len(self.index),len(self.columns),len(self.depths),\
                self.index[0],self.index[-1],
                self.columns[0],self.columns[-1],
                self.depths[0],self.depths[-1])
        return string

    __repr__ = __str__

    def head(self,n=5):
        df = self.to_2D()
        return df.head(n)

    def tail(self,n=5):
        df = self.to_2D()
        return df.tail(n)

    def to_2D(self):
        df = pd.DataFrame(np.row_stack(self.values),columns=self.depths,index=\
                pd.MultiIndex.from_product([self.index,self.columns]))
        return df

    def fillna(self,value=0):
        self.values[np.isnan(self.values)] = value

    @property
    def shape(self):
        return len(self.index),len(self.columns),len(self.depths)

    def __getitem__(self,slices):
        si,sc,sd = [self._parse_slice_(sl,ll) for sl,ll in \
                zip(slices,[self.index,self.columns,self.depths])]
        values = self.values[si,:,:][:,sc,:][:,:,sd]
        index = list_index(self.index,si)
        columns = list_index(self.columns,sc)
        depths = list_index(self.depths,sd)
        return DataFrame3D(values,index,columns,depths)

    def __setitem__(self,slices,arr3d:np.array):
        si,sc,sd = [self._parse_slice_(sl,ll) for sl,ll in \
                zip(slices,[self.index,self.columns,self.depths])]
        self.values[si[:,np.newaxis,np.newaxis],sc[np.newaxis,:,np.newaxis],\
                sd[np.newaxis,np.newaxis,:]] = arr3d

    def _parse_slice_(self,sl,ll):
        if isinstance(sl,slice):
            start = sl.start if (sl.start is None) or (isinstance(sl.start,int)) else ll.index(sl.start)
            stop = sl.stop if (sl.stop is None) or (isinstance(sl.stop,int)) else ll.index(sl.stop)
            sl = slice(start,stop,sl.step)
            idx = np.array([*range(len(ll))][sl])
        elif isinstance(sl,list):
            idx = index_lshort_in_llong(sl,ll)
        elif isinstance(sl,str):
            idx = np.array([ll.index(sl)])
        return idx

def readm2df_3d(name:str)->pd.DataFrame:
    [data,dates,ids,times] = read_binary_array_3d(name)
    df = DataFrame3D(data,dates,ids,times)
    return df

#---------------------- Rq Tick Data ----------------------
def rq2qp_ids(dt:str=None)->pd.Series:
    if dt is None: dt = today()
    qp_ids_pd = all_ids_types_pd()
    types = qp_ids_pd.drop_duplicates()
    qp_ids = pd.concat([globals()[tp]().tickers_rq(dt) for _,tp in types.items()])
    qp_ids = pd.Series(qp_ids,index=all_ids())
    return qp_ids

def rq_raw_ids_df(type:str='CS')->pd.DataFrame:
    '''
    type: CS/ETF/INDX/Future/Option
    '''
    file = os.path.join(gvars.RqDir,'tickers',type+'.csv')
    df = pd.read_csv(file,index_col='order_book_id')
    return df

def rq_raw_ids(type:str='CS')->list:
    df = rq_raw_ids_df(type)
    ids = df.index.tolist()
    ids.sort()
    return ids

def n_rq_raw_ids(type:str='CS')->int:
    df = rq_raw_ids_df(type)
    return len(df)

def rq_types_mapping(types:list=None)->dict:
    if types is None:
        # Find all of subclasses of Instruments()
        types = [instr.__name__ for instr in Instruments.__subclasses__()]
    return {type:globals()[type]().rq_type for type in types}

def read_tick_data_file(file:str):
    data = pd.read_csv(file,compression='gzip',error_bad_lines=False,index_col=0)
    data = data[data.index.notnull()]
    return data

def read_mb1_data_file(file:str):
    data = pd.read_csv(file,compression='gzip',error_bad_lines=False,index_col=0)
    data = data[data.index.notnull()]
    data.index = data.index.astype(int)
    return data

#%%
if __name__=='__main__':
    print(yesterday())
    # Usage for DataFrame3D
    #values = np.random.randn(4,3,4)
    #index = [str(n) for n in range(4)]
    #columns = [str(n) for n in range(3)]
    #depths = [str(n) for n in range(4)]
    #df3 = DataFrame3D(1,index,columns,depths)
    ##print(df3[:'1','1',['0','2']].to_2D())
    ##print(df3[:1,'1',['0','2']].to_2D())
    #print(df3[:,'1',['0','2']].to_2D())
    #print(df3[::2,'1',['0','2']].to_2D())
    #print(df3[1:3,'1',['0','2']].to_2D())
    #print(df3)
    #print(df3.head())
    #df3 = DataFrame3D(index=index,columns=columns,depths=depths)
    #print(df3)
    #print(df3.head())
    #df3.fillna(0)
    #print(df3.head())
    #df3 = DataFrame3D(values,index,columns,depths)
    #print(df3.head())
    #print(df3[:,'1',['0','2']])
    #print(df3.values)
    #df3[['0'],:,:2] = np.ones([1,3,2])
    #print(df3.values)
    #new_index = ['1','2','4'] 
    #new_columns = ['1','2','4'] 
    #new_df3 = DataFrame3D(df3,index=new_index,columns=new_columns)
    #print(new_df3)
    #print(new_df3.head())

    #readm2df_3d('/qp/data/tmp/mb/b15/high.b15.bin')
    #z = readm2df_3d('/qp/data/tmp/mb/b30/high.b30.bin')
    #print(z)
    #print(z.head(30))
    #print(read_mb1_data_file('/qp/data/tmp/rq/csv/ashare/mb1/20190102/000001.tgz').head())
    #read_tick_data_file('/qp/data/tmp/rq/raw/tick/INDX/20180817/000300.XSHG.tgz')
    #print(rq2qp_ids())
    #ss= stock()
    #print(ss.tickers_rq(20130101))
    #ii = index()
    #print(ii.tickers_rq(20130101))
