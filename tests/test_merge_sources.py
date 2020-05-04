# coding=utf-8

import unittest

import numpy as np
import pandas as pd

from clustermatch.utils.data import merge_sources
from .utils import get_data_file


class ReadTomateTest(unittest.TestCase):
    def test_merge_sources_using_ps(self):
        ## Preparar
        data_file = get_data_file('ps_2011_2012.csv')

        ## Correr
        ps_pro = merge_sources(data_file)[0]

        ## Validar
        assert ps_pro is not None
        assert hasattr(ps_pro, 'shape')
        assert ps_pro.shape[0] == 10
        assert ps_pro.shape[1] == 13

        assert ps_pro.notnull().all().all()

        # arriba izquierda
        assert ps_pro.round(3).loc['Arom-1', '552'] == 0.000
        assert ps_pro.round(3).loc['Arom-1', '553'] == 0.000
        assert ps_pro.round(3).loc['Arom-5', '552'] == 0.533

        # arriba derecha
        assert ps_pro.round(3).loc['Arom-1', 'Bigua'] == 0.111
        assert ps_pro.round(3).loc['Arom-1', 'Elpida'] == 0.037
        assert ps_pro.round(3).loc['Arom-5', 'Elpida'] == 0.296

        # abajo derecha
        assert ps_pro.round(3).loc['Jug-4', 'Bigua'] == 0.172
        assert ps_pro.round(3).loc['Jug-4', 'Elpida'] == 0.586
        assert ps_pro.round(3).loc['Jug-1', 'Elpida'] == 0.000

        # abajo izquierda
        assert ps_pro.round(3).loc['Jug-4', '553'] == 0.158
        assert ps_pro.round(3).loc['Jug-4', '552'] == 0.533
        assert ps_pro.round(3).loc['Jug-1', '552'] == 0.000

    def test_merge_sources_using_vo(self):
        ## Preparar
        data_file = get_data_file('vo_2011_2012.csv')

        ## Correr
        vo_pro = merge_sources(data_file)[0]

        ## Validar
        assert vo_pro is not None
        assert hasattr(vo_pro, 'shape')
        assert vo_pro.shape[0] == 42
        assert vo_pro.shape[1] == 11

        assert vo_pro.notnull().all().all()

        # arriba izquierda
        assert vo_pro.round(3).loc['UNK 43', '552'] == 5.12
        assert vo_pro.round(3).loc['UNK 43', '553'] == 4.77
        assert vo_pro.round(3).loc['3mBUTANAL', '552'] == 0.000

        # arriba derecha
        assert vo_pro.round(3).loc['UNK 43', 'Bigua'] == 2.43
        assert vo_pro.round(3).loc['UNK 43', 'Elpida'] == 3.40
        assert vo_pro.round(3).loc['3mBUTANAL', 'Elpida'] == 1.34

        # abajo derecha
        assert vo_pro.round(3).loc['TRANS2HEXENAL', 'Bigua'] == 0.00
        assert vo_pro.round(3).loc['TRANS2HEXENAL', 'Elpida'] == 7.11
        assert vo_pro.round(3).loc['CIS2HEXENAL', 'Elpida'] == 0.00

        # abajo izquierda
        assert vo_pro.round(3).loc['TRANS2HEXENAL', '553'] == 6.90
        assert vo_pro.round(3).loc['TRANS2HEXENAL', '552'] == 5.40
        assert vo_pro.round(3).loc['CIS2HEXENAL', '552'] == 0.000

    def test_merge_sources_using_me_with_rep_merge_mean(self):
        ## Preparar
        data_file = get_data_file('me_2011_2012.csv')

        ## Correr
        me_pro = merge_sources(data_file, rep_merge=np.mean)[0]

        ## Validar
        assert me_pro is not None
        assert hasattr(me_pro, 'shape')
        assert me_pro.shape[0] == 89
        assert me_pro.shape[1] == 44

        # chequear todos los valores nulos
        assert pd.isnull(me_pro.loc['NA_2106.37', '3806'])
        assert pd.isnull(me_pro.loc['NA_1608.87', '3815'])
        assert pd.isnull(me_pro.loc['NA_2106.37', '4748'])
        assert pd.isnull(me_pro.loc['Glucoheptonic acid-1.4-lactone', '4748'])
        assert pd.isnull(me_pro.loc['NA_2106.37', '560'])
        assert pd.isnull(me_pro.loc['Glucoheptonic acid-1.4-lactone', '560'])

        # arriba izquierda
        assert me_pro.round(3).loc['serine', '549'] == 19.905
        assert me_pro.round(3).loc['serine', '551'] == 13.735

        # arriba derecha
        assert me_pro.round(3).loc['serine', '4751'] == 38.439
        assert me_pro.round(3).loc['Ethanolamine', '4751'] == 1.619

        # abajo izquierda
        assert me_pro.round(3).loc['Sucrose', '549'] == 171.211
        assert me_pro.round(3).loc['NA_2627.66', '549'] == 3.853

        # abajo derecha
        assert me_pro.round(3).loc['NA_2627.66', '4751'] == 5.018
        assert me_pro.round(3).loc['NA_2627.66', '4750'] == 13.353

    def test_merge_sources_using_ag(self):
        ## Preparar
        data_file = get_data_file('ag_2011_2012.csv')

        ## Correr
        ag_pro = merge_sources(data_file)[0]

        ## Validar
        assert ag_pro is not None
        assert hasattr(ag_pro, 'shape')
        assert ag_pro.shape[0] == 16
        assert ag_pro.shape[1] == 19

        # chequear todos los valores nulos
        # assert pd.isnull(ag_pro.loc['perim', '549'])

        # arriba izquierda
        assert ag_pro.round(3).loc['peso', '549'] == 287.247
        assert ag_pro.round(3).loc['peso', '550'] == 189.247
        assert ag_pro.round(3).loc['perim', '549'] == 280.336

        # arriba derecha
        assert ag_pro.round(3).loc['peso', '572'] == 10.31
        assert ag_pro.round(3).loc['firmeza', '572'] == 1.383

        # abajo izquierda
        assert ag_pro.round(3).loc['a_cielab', '549'] == 44.870
        assert ag_pro.round(3).loc['b_cielab', '549'] == 61.691

        # abajo derecha
        assert ag_pro.round(3).loc['b_cielab', '572'] == 57.386
        assert ag_pro.round(3).loc['b_cielab', '571'] == 61.842

        # Casos especiales
        # todos ceros
        assert ag_pro.round(3).loc['area_indent', '572'] == 0.000

        # valores cercanos a cero
        assert ag_pro.round(3).loc['area_indent', '571'] == 0.038

    def test_merge_sources_using_ap(self):
        ## Preparar
        data_file = get_data_file('ap_2011_2012.csv')

        ## Correr
        ap_pro = merge_sources(data_file)[0]

        ## Validar
        assert ap_pro is not None
        assert hasattr(ap_pro, 'shape')
        assert ap_pro.shape[0] == 7
        assert ap_pro.shape[1] == 42

        # chequear todos los valores nulos
        # assert pd.isnull(ag_pro.loc['perim', '549'])

        # arriba izquierda
        assert ap_pro.round(3).loc['Peso', '549'] == 0.532
        assert ap_pro.round(3).loc['Peso', '550'] == 0.620

        # arriba derecha
        assert ap_pro.round(3).loc['Peso', 'elpida'] == 0.540
        assert ap_pro.round(3).loc['TEAC HID (meq. Trolox %)', 'elpida'] == 0.351

        # abajo izquierda
        assert ap_pro.round(3).loc['carotenos (mg%)', '549'] == 0.260
        assert ap_pro.round(3).loc['LICOP (mg%)', '549'] == 3.969

        # abajo derecha
        assert ap_pro.round(3).loc['carotenos (mg%)', 'elpida'] == 0.511
        assert ap_pro.round(3).loc['carotenos (mg%)', 'bigua'] == 0.319

        # Casos especiales
        # un nan en el medio
        assert ap_pro.round(3).loc['TEAC LIP (meq. Trolox %)', '558'] == 0.029

    def test_merge_sources_index_name(self):
        ## Preparar
        data_file = get_data_file('ap_2011_2012.csv')

        ## Correr
        ap_pro = merge_sources(data_file)[0]

        ## Validar
        assert ap_pro is not None
        assert hasattr(ap_pro, 'index')
        assert ap_pro.index.name == 'features'

    def test_merge_source_returning_names_using_ag(self):
        ## Preparar
        data_file = get_data_file('ag_2011_2012.csv')

        ## Correr
        ag_pro, ag_nom, _ = merge_sources(data_file)

        ## Validar
        assert ag_pro is not None

        assert ag_nom is not None
        assert len(ag_nom) == 16
        assert ag_nom[0] == 'peso'
        assert ag_nom[1] == 'firmeza'
        assert ag_nom[7] == 'area_indent'
        assert ag_nom[14] == 'a_cielab'
        assert ag_nom[15] == 'b_cielab'

    def test_merge_source_returning_names_using_ap(self):
        ## Preparar
        data_file = get_data_file('ap_2011_2012.csv')

        ## Correr
        ap_pro, ap_nom, _ = merge_sources(data_file)

        ## Validar
        assert ap_pro is not None

        assert ap_nom is not None
        assert len(ap_nom) == 7
        assert ap_nom[0] == 'Peso'
        assert ap_nom[1] == 'TEAC HID (meq. Trolox %)'
        assert ap_nom[2] == 'TEAC LIP (meq. Trolox %)'
        assert ap_nom[3] == 'FRAP (meq. Trolox %)'
        assert ap_nom[4] == 'FOLIN (mg Ac Galico/100g)'
        assert ap_nom[5] == 'LICOP (mg%)'
        assert ap_nom[6] == 'carotenos (mg%)'

    def test_merge_source_returning_names_using_ap_ps(self):
        ## Preparar
        data_files = [get_data_file('ap_2011_2012.csv'),
                      get_data_file('ps_2011_2012.csv')]

        ## Correr
        pro, nom, _ = merge_sources(data_files)

        ## Validar
        assert pro is not None

        assert nom is not None
        assert len(nom) == 7 + 10

        ap_var_names = ['Peso', 'TEAC HID (meq. Trolox %)', 'TEAC LIP (meq. Trolox %)',
                        'FRAP (meq. Trolox %)', 'FOLIN (mg Ac Galico/100g)', 'LICOP (mg%)',
                        'carotenos (mg%)']
        if not (ap_var_names == nom[:7] or ap_var_names == nom[-7:]):
            self.fail('ap variables not found')

        ps_var_names = ['Arom-1', 'Arom-5', 'Sab-1', 'Sab-5', 'Dulz-1', 'Dulz-5', 'Acid-1',
                        'Acid-5', 'Jug-1', 'Jug-4']
        if not (ps_var_names == nom[:10] or ps_var_names == nom[-10:]):
            self.fail('ap variables not found')

    def test_merge_source_returning_sources_using_ap_ps(self):
        ## Preparar
        data_files = [get_data_file('ap_2011_2012.csv'),
                      get_data_file('ps_2011_2012.csv')]

        ## Correr
        pro, nom, sources = merge_sources(data_files)

        ## Validar
        assert pro is not None
        assert nom is not None

        assert sources is not None
        assert len(sources) == 7 + 10
        assert len(set(sources)) == 2  # unique source names
        assert 'ps_2011_2012' in sources
        assert 'ap_2011_2012' in sources

        if sources[0] == 'ps_2011_2012':
            assert len(set(sources[:10])) == 1
            assert 'ps_2011_2012' in set(sources[:10])

            assert len(set(sources[-7:])) == 1
            assert 'ap_2011_2012' in set(sources[-7:])
        else:
            assert len(set(sources[:7])) == 1
            assert 'ap_2011_2012' in set(sources[:7])

            assert len(set(sources[-10:])) == 1
            assert 'ps_2011_2012' in set(sources[-10:])

    def test_merge_sources_multiple_using_ps_vo(self):
        ## Preparar
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        fuentes = [ps_data_file, vo_data_file]

        ## Correr
        procesado, nombres, _ = merge_sources(fuentes)

        ## Validar
        assert procesado is not None
        assert hasattr(procesado, 'shape')
        assert procesado.shape[0] == 10 + 42
        assert procesado.shape[1] == 13  # columnas totales, se cuenta una sola vez las compartidas

        # ps
        assert procesado.round(3).loc['Arom-1', '552'] == 0.00
        assert procesado.round(3).loc['Arom-1', '3837'] == 0.00
        assert procesado.round(3).loc['Arom-1', '4735'] == 0.063
        assert procesado.round(3).loc['Arom-1', '1589'] == 0.231
        assert procesado.round(3).loc['Arom-1', 'Bigua'] == 0.111
        assert procesado.round(3).loc['Arom-1', 'Elpida'] == 0.037

        assert procesado.round(3).loc['Jug-4', '552'] == 0.533  # abajo izquierda
        assert procesado.round(3).loc['Jug-4', 'Elpida'] == 0.586  # abajo derecha

        # vo
        assert procesado.round(3).loc['UNK 43', '552'] == 5.12
        assert procesado.round(3).loc['UNK 43', '3837'] == 3.98
        assert pd.isnull(procesado.round(3).loc['UNK 43', '4735'])
        assert pd.isnull(procesado.round(3).loc['UNK 43', '1589'])
        assert procesado.round(3).loc['UNK 43', 'Bigua'] == 2.430
        assert procesado.round(3).loc['UNK 43', 'Elpida'] == 3.400

        assert procesado.round(3).loc['TRANS2HEXENAL', '552'] == 5.400  # abajo izquierda
        assert procesado.round(3).loc['TRANS2HEXENAL', 'Elpida'] == 7.110  # abajo derecha

    def test_merge_sources_multiple_using_me_ag(self):
        ## Preparar
        me_data_file = get_data_file('me_2011_2012.csv')
        ag_data_file = get_data_file('ag_2011_2012.csv')

        fuentes = [me_data_file, ag_data_file]

        ## Correr
        procesado, nombres, _ = merge_sources(fuentes)

        ## Validar
        assert procesado is not None
        assert hasattr(procesado, 'shape')
        assert procesado.shape[0] == 89 + 16
        assert procesado.shape[1] == 47  # columnas totales, se cuenta una sola vez las compartidas

        # me
        ## valores nulos
        assert pd.isnull(procesado.loc['NA_2106.37', '3806'])
        assert pd.isnull(procesado.loc['NA_1608.87', '3815'])
        assert pd.isnull(procesado.loc['NA_2106.37', '4748'])
        assert pd.isnull(procesado.loc['Glucoheptonic acid-1.4-lactone', '4748'])
        assert pd.isnull(procesado.loc['NA_2106.37', '560'])
        assert pd.isnull(procesado.loc['Glucoheptonic acid-1.4-lactone', '560'])
        ## arriba izquierda
        assert procesado.round(3).loc['serine', '549'] == 19.905
        assert procesado.round(3).loc['serine', '551'] == 13.735
        ## arriba derecha
        assert procesado.round(3).loc['serine', '4751'] == 38.439
        assert procesado.round(3).loc['Ethanolamine', '4751'] == 1.619
        ## abajo izquierda
        assert procesado.round(3).loc['Sucrose', '549'] == 171.211
        assert procesado.round(3).loc['NA_2627.66', '549'] == 3.853
        ## abajo derecha
        assert procesado.round(3).loc['NA_2627.66', '4751'] == 5.018
        assert procesado.round(3).loc['NA_2627.66', '4750'] == 13.353

        # ag
        ## arriba izquierda
        assert procesado.round(3).loc['peso', '549'] == 287.247
        assert procesado.round(3).loc['peso', '550'] == 189.247
        assert procesado.round(3).loc['perim', '549'] == 280.336
        ## arriba derecha
        assert procesado.round(3).loc['peso', '572'] == 10.31
        assert procesado.round(3).loc['firmeza', '572'] == 1.383
        ## abajo izquierda
        assert procesado.round(3).loc['a_cielab', '549'] == 44.870
        assert procesado.round(3).loc['b_cielab', '549'] == 61.691
        ## abajo derecha
        assert procesado.round(3).loc['b_cielab', '572'] == 57.386
        assert procesado.round(3).loc['b_cielab', '571'] == 61.842
        ## todos ceros
        assert procesado.round(3).loc['area_indent', '572'] == 0.000
        ## valores cercanos a cero
        assert procesado.round(3).loc['area_indent', '571'] == 0.038

    def test_merge_sources_xls_2008_2009(self):
        ## Correr
        procesado = merge_sources(get_data_file('2008-2009.xls'))[0]

        ## Validar
        assert procesado is not None
        assert hasattr(procesado, 'shape')
        assert procesado.shape[0] == 101 + 26 + 29
        # assert procesado.shape[1] == 47  # columnas totales, se cuenta una sola vez las compartidas

        # volátiles
        ## valores nulos
        assert pd.isnull(procesado.loc['4-metil-3-hepten-2-ona', '569'])
        assert pd.isnull(procesado.loc['4-metil-3-hepten-2-ona', '3806'])
        assert pd.isnull(procesado.loc['1,4-pentadieno', '572'])
        assert pd.isnull(procesado.loc['1,4-pentadieno', '3842'])
        assert pd.isnull(procesado.loc['1,4-pentadieno', '4618'])
        assert pd.isnull(procesado.loc['UNK m/z 83', '571'])
        assert pd.isnull(procesado.loc['UNK m/z 83', '560']) # 560 no está en volátiles
        ## arriba izquierda
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '552'] == 0.709
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '557'] == 0.612
        assert procesado.round(3).loc['1,4-pentadieno', '552'] == 0.635
        ## arriba derecha
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '4750'] == 62.239
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '4739'] == 0.596
        assert procesado.round(3).loc['1,4-pentadieno', '4750'] == 0.331
        ## abajo derecha
        assert procesado.round(3).loc['verdil acetato', '4750'] == 10.872
        assert procesado.round(3).loc['verdil acetato', '4739'] == 8.20
        assert procesado.round(3).loc['UNK m/z 95', '4750'] == 4.747
        ## abajo izquierda
        assert procesado.round(3).loc['UNK m/z 95', '552'] == 6.866
        assert procesado.round(3).loc['verdil acetato', '552'] == 17.43
        assert procesado.round(3).loc['verdil acetato', '557'] == 11.522

        # NMR
        ## valores nulos
        assert pd.isnull(procesado.loc['Galactose', '552'])
        assert pd.isnull(procesado.loc['Xylose', '552'])
        assert pd.isnull(procesado.loc['Galactose', 'LA407'])  # 560 no está en NMR
        ## arriba izquierda
        assert procesado.round(3).loc['Citrate', '4750'] == 55.392
        assert procesado.round(3).loc['Fructose', '4750'] == 174.413
        assert procesado.round(3).loc['Citrate', '565'] == 35.852
        ## arriba derecha
        assert procesado.round(3).loc['Citrate', '4739'] == 20.354
        assert procesado.round(3).loc['Citrate', '569'] == 23.565
        assert procesado.round(3).loc['Fructose', '4739'] == 248.168
        ## abajo derecha
        assert procesado.round(3).loc['Galactose', '4739'] == 26.674
        assert procesado.round(3).loc['Xylose', '4739'] == 13.535
        assert procesado.round(3).loc['Xylose', '569'] == 9.728
        ## abajo izquierda
        assert procesado.round(3).loc['UNK m/z 95', '552'] == 6.866

        # Antioxidantes
        ## valores nulos
        assert pd.isnull(procesado.loc['Caffeoylhexaric acid', '569'])
        assert pd.isnull(procesado.loc['Trihydroxycinnamoylquinic acid', '569'])
        assert pd.isnull(procesado.loc['SR', '569'])
        ## arriba izquierda
        assert procesado.round(3).loc['Total polyphenols', '552'] == 65.957
        assert procesado.round(3).loc['Antioxidant capacity (FRAP)', '552'] == 0.269
        assert procesado.round(3).loc['Total polyphenols', '557'] == 57.728
        ## arriba derecha
        assert procesado.round(3).loc['Total polyphenols', '3815'] == 68.341
        assert procesado.round(3).loc['Total polyphenols', 'LA407'] == 147.041
        assert procesado.round(3).loc['Antioxidant capacity (FRAP)', '3815'] == 0.277
        ## abajo derecha
        assert procesado.round(3).loc['ED50', '3815'] == 34.037
        assert procesado.round(3).loc['SR', '3815'] == 14.587
        assert procesado.round(3).loc['SR', 'LA407'] == 19.137
        ## abajo izquierda
        assert procesado.round(3).loc['ED50', '552'] == 0.277
        assert procesado.round(3).loc['SR', '552'] == 11.367
        assert procesado.round(3).loc['SR', '557'] == 15.357
        ## casos especiales
        assert procesado.round(3).loc['Total Tocopherol', '569'] == 1129.201

    def test_merge_source_returning_sources_using_xls_2008_2009(self):
        ## Correr
        pro, nom, sources = merge_sources(get_data_file('2008-2009.xlsx'))

        ## Validar
        assert pro is not None
        assert nom is not None

        assert sources is not None
        assert len(sources) == 101 + 26 + 29
        assert len(set(sources)) == 3  # unique source names

        assert 'Volátiles' in sources
        assert 'NMR' in sources
        assert 'Antioxidantes' in sources

        volatiles_start_idx = sources.index('Volátiles')
        volatiles_range = slice(volatiles_start_idx, volatiles_start_idx + 101)
        assert len(set(sources[volatiles_range])) == 1
        assert 'Volátiles' in set(sources[volatiles_range])

        nmr_start_idx = sources.index('NMR')
        nmr_range = slice(nmr_start_idx, nmr_start_idx + 26)
        assert len(set(sources[nmr_range])) == 1
        assert 'NMR' in set(sources[nmr_range])

        antio_start_idx = sources.index('Antioxidantes')
        antio_range = slice(antio_start_idx, antio_start_idx + 29)
        assert len(set(sources[antio_range])) == 1
        assert 'Antioxidantes' in set(sources[antio_range])

    def test_merge_source_repeated_features_same_source(self):
        ## Run
        try:
            merge_sources(get_data_file('2008-2009_repeated_feature_names_same_source.xls'))
            self.fail('It should have failed')
        except ValueError as e:
            # check that the message contains the word 'duplicated'
            assert 'duplicated' in e.args[0]

    def test_merge_source_repeated_features_different_source(self):
        ## Correr
        pro, nom, sources = merge_sources(get_data_file('2008-2009_repeated_feature_names_different_source.xls'))

        ## Validar
        assert pro is not None
        assert len(pro) == 5 + 6 + 12

        assert nom is not None
        assert len(nom) == 5 + 6 + 12

        assert sources is not None
        assert len(sources) == 5 + 6 + 12

        assert pro.index.is_unique
        assert '1,4-pentadieno' not in pro.index

        # antioxidantes variable
        key01 = '1,4-pentadieno (Antioxidantes)'
        if key01 not in pro.index:
            self.fail(key01 + ' not found')

        assert pro.round(3).loc[key01, '552'] == 0.377
        assert pro.round(3).loc[key01, '557'] == 0.251
        assert pro.round(3).loc[key01, '560'] == 0.000
        assert pro.round(3).loc[key01, '565'] == 0.185
        assert pro.round(3).loc[key01, '569'] == 0.395

        # volátiles variable
        key02 = '1,4-pentadieno (Volátiles)'
        if key02 not in pro.index:
            self.fail(key02 + ' not found')

        assert pro.round(3).loc[key02, '552'] == 0.151
        assert pro.round(3).loc[key02, '557'] == 0.282
        assert pro.round(3).loc[key02, '565'] == 0.097
        assert pro.round(3).loc[key02, '569'] == 3.268

    def test_merge_source_single_zip_flat_and_single_source(self):
        ## Preparar
        data_file = get_data_file('ap_2011_2012.zip')

        ## Correr
        ap_pro, ap_nom, ap_sources = merge_sources(data_file)

        ## Validar
        assert ap_pro is not None
        assert hasattr(ap_pro, 'shape')
        assert ap_pro.shape[0] == 7
        assert ap_pro.shape[1] == 42

        assert ap_nom is not None
        assert len(ap_nom) == 7
        ap_var_names = ['Peso', 'TEAC HID (meq. Trolox %)', 'TEAC LIP (meq. Trolox %)',
                        'FRAP (meq. Trolox %)', 'FOLIN (mg Ac Galico/100g)', 'LICOP (mg%)',
                        'carotenos (mg%)']
        assert ap_var_names == ap_nom

        assert ap_sources is not None
        assert len(ap_sources) == 7
        assert len(set(ap_sources)) == 1  # unique source names
        assert 'ap_2011_2012' in ap_sources

        # chequear todos los valores nulos
        # assert pd.isnull(ag_pro.loc['perim', '549'])

        # arriba izquierda
        assert ap_pro.round(3).loc['Peso', '549'] == 0.532
        assert ap_pro.round(3).loc['Peso', '550'] == 0.620

        # arriba derecha
        assert ap_pro.round(3).loc['Peso', 'elpida'] == 0.540
        assert ap_pro.round(3).loc['TEAC HID (meq. Trolox %)', 'elpida'] == 0.351

        # abajo izquierda
        assert ap_pro.round(3).loc['carotenos (mg%)', '549'] == 0.260
        assert ap_pro.round(3).loc['LICOP (mg%)', '549'] == 3.969

        # abajo derecha
        assert ap_pro.round(3).loc['carotenos (mg%)', 'elpida'] == 0.511
        assert ap_pro.round(3).loc['carotenos (mg%)', 'bigua'] == 0.319

        # Casos especiales
        # un nan en el medio
        assert ap_pro.round(3).loc['TEAC LIP (meq. Trolox %)', '558'] == 0.029

    def test_merge_source_single_zip_upper_cased_filename(self):
        ## Preparar
        data_file = get_data_file('ap_2011_2012_2.ZIP')

        ## Correr
        ap_pro, ap_nom, ap_sources = merge_sources(data_file)

        ## Validar
        assert ap_pro is not None
        assert hasattr(ap_pro, 'shape')
        assert ap_pro.shape[0] == 7
        assert ap_pro.shape[1] == 42

        assert ap_nom is not None
        assert len(ap_nom) == 7
        ap_var_names = ['Peso', 'TEAC HID (meq. Trolox %)', 'TEAC LIP (meq. Trolox %)',
                        'FRAP (meq. Trolox %)', 'FOLIN (mg Ac Galico/100g)', 'LICOP (mg%)',
                        'carotenos (mg%)']
        assert ap_var_names == ap_nom

        assert ap_sources is not None
        assert len(ap_sources) == 7
        assert len(set(ap_sources)) == 1  # unique source names
        assert 'ap_2011_2012' in ap_sources

    def test_merge_source_csv_and_zip_flat_with_multiple_source(self):
        # Preparar
        data_files = [get_data_file('me_2011_2012.csv'),
                      get_data_file('2008-2009_en_dos.zip')]

        ## Correr
        procesado, nombres, sources = merge_sources(data_files)

        ## Validar
        assert procesado is not None
        assert hasattr(procesado, 'shape')
        assert procesado.shape[0] == 89 + 101 + 26 + 29

        assert nombres is not None
        assert len(nombres) == 89 + 101 + 26 + 29

        assert sources is not None
        assert len(sources) == 89 + 101 + 26 + 29
        assert len(set(sources)) == 4
        assert 'Volátiles' in sources
        assert 'NMR' in sources
        assert 'Antioxidantes' in sources
        assert 'me_2011_2012' in sources

        # me, algunos valores
        assert procesado.round(3).loc['serine', '549'] == 19.905
        assert procesado.round(3).loc['serine', '4751'] == 38.439

        # Volátiles, algunos valores
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '552'] == 0.709
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '4750'] == 62.239

        # NMR, algunos valores
        assert procesado.round(3).loc['Citrate', '4750'] == 55.392
        assert procesado.round(3).loc['Citrate', '565'] == 35.852

        # Antioxidantes, algunos valores
        assert procesado.round(3).loc['Total polyphenols', '552'] == 65.957
        assert procesado.round(3).loc['Total polyphenols', '3815'] == 68.341

    def test_merge_source_csv_and_zip_nonflat_with_multiple_source(self):
        # Preparar
        data_files = [get_data_file('me_2011_2012.csv'),
                      get_data_file('2008-2009_en_dos_nonflat.zip')]

        ## Correr
        procesado, nombres, sources = merge_sources(data_files)

        ## Validar
        assert procesado is not None
        assert hasattr(procesado, 'shape')
        assert procesado.shape[0] == 89 + 101 + 26 + 29

        assert nombres is not None
        assert len(nombres) == 89 + 101 + 26 + 29

        assert sources is not None
        assert len(sources) == 89 + 101 + 26 + 29
        assert len(set(sources)) == 4
        assert 'Volátiles' in sources
        assert 'NMR' in sources
        assert 'Antioxidantes' in sources
        assert 'me_2011_2012' in sources

        # me, algunos valores
        assert procesado.round(3).loc['serine', '549'] == 19.905
        assert procesado.round(3).loc['serine', '4751'] == 38.439

        # Volátiles, algunos valores
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '552'] == 0.709
        assert procesado.round(3).loc['4-metil-3-hepten-2-ona', '4750'] == 62.239

        # NMR, algunos valores
        assert procesado.round(3).loc['Citrate', '4750'] == 55.392
        assert procesado.round(3).loc['Citrate', '565'] == 35.852

        # Antioxidantes, algunos valores
        assert procesado.round(3).loc['Total polyphenols', '552'] == 65.957
        assert procesado.round(3).loc['Total polyphenols', '3815'] == 68.341

    def test_read_same_object_in_different_sources(self):
        # prepare
        data00 = merge_sources(get_data_file('sampleAA.xlsx'))[0]
        data01 = merge_sources(get_data_file('sampleBB.xlsx'))[0]
        # data00 = merge_sources([get_data_file('sampleAA_NMR.csv'), get_data_file('sampleAA_Agronomics.csv')])[0]
        # data01 = merge_sources([get_data_file('sampleBB_VAgronomics.csv'), get_data_file('sampleBB_NMR.csv')])[0]

        # validate
        assert data00 is not None
        assert data01 is not None

        assert data00.shape == data01.shape
        assert data00.index.tolist() != data01.index.tolist()

        data00_s_index = sorted(data00.index.tolist())
        data01_s_index = sorted(data01.index.tolist())

        assert data00_s_index == data01_s_index

        assert data00.columns.tolist() == data01.columns.tolist()

        equals = (data00.loc[data00_s_index, data00.columns] == data01.loc[data01_s_index, data01.columns]) | \
                 (np.isnan(data00.loc[data00_s_index, data00.columns]) & \
                  np.isnan(data01.loc[data01_s_index, data01.columns]))

        assert equals.all().all()

    def test_merge_sources_categorical_data(self):
        ## Preparar
        data_file = get_data_file('categorical.xlsx')

        ## Correr
        ps_pro = merge_sources(data_file)[0]

        ## Validar
        assert ps_pro is not None
        assert hasattr(ps_pro, 'shape')
        assert ps_pro.shape[0] == 10 + 26
        assert ps_pro.shape[1] == 22, ps_pro.shape

        # numerical
        assert round(ps_pro.round(3).loc['GABA', '4618'], 3) == 3.669

        # abajo izquierda
        assert ps_pro.round(3).loc['Flavor', '552'] == 'Caract.'
        assert ps_pro.round(3).loc['Arom', '552'] == 'Other'
        assert round(ps_pro.round(3).loc['Shape', '552'], 3) == 3.769
