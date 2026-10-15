"""Transfer integrity: padding, group isolation, train-only fitting and source rows."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ips.data import ROOT, feature_transform
from ips.data_access.supplied import SuppliedFingerprints, group_split, read_supplied
from ips.algorithms.kan.supplied_experiment import PaddedDataset, source_settings, measure
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models.reference import fit_reference_points
from ips.algorithms.kan.models.refinement import RefinedKANMixer, db_features
from ips.algorithms.kan.runtime import configure_runtime
from types import SimpleNamespace


class SuppliedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_runtime(42, "cpu", 2)
        (ROOT / 'runs').mkdir(exist_ok=True)

    def data(self):
        n = 40
        xy = np.repeat(np.column_stack([np.arange(10), np.arange(10)*2]),4,axis=0).astype(float)
        return SuppliedFingerprints("dlulib", np.tile([-100,-60,-50],(n,1)),xy,
            np.ones(n,dtype=int),np.zeros(n,dtype=int),np.full(n,"raw.xlsx"),np.arange(2,n+2),
            np.repeat(np.arange(10),4),["AP1","AP2","AP3"],-100,[],[])

    def test_padding_preserves_real_signals_and_absent_mask(self):
        raw=self.data(); prep={"coordinates":{"center":[0,0],"shared_scale":2}}
        data=PaddedDataset(raw,np.arange(len(raw)),prep,[{"class_id":0,"building":0,"floor":1}],"train")
        self.assertEqual(tuple(data.features.shape),(40,520,2))
        self.assertTrue((data.features[:,3:]==0).all())
        self.assertTrue((data.features[:,0]==0).all())
        torch.testing.assert_close(data.features[:,1:3],torch.from_numpy(feature_transform(raw.rssi[:,1:3])))
        self.assertEqual(data.source_rows.tolist(),list(range(2,42)))
        np.testing.assert_allclose(data.xy.numpy(),raw.xy/2)
        self.assertTrue((db_features(data.features)[:,3:]==0).all())

    def test_group_split_keeps_every_repeat_out_of_other_split(self):
        raw=self.data(); train,val=group_split(raw)
        self.assertEqual(len(train),32);self.assertEqual(len(val),8)
        self.assertEqual(set(raw.rp[train]) & set(raw.rp[val]),set())
        np.testing.assert_array_equal(group_split(raw)[0],train)
        np.testing.assert_array_equal(np.sort(np.r_[train,val]),np.arange(40))

    def test_reference_catalogue_never_consumes_heldout_positions(self):
        # No local historical runs are required for these model contract tests.
        import json
        config=json.loads((ROOT/'src/ips/algorithms/kan/configs/accuracy_spatial.json').read_text())
        config['mixer'].update(channels=8,blocks=1,token_hidden=8,channel_hidden=8,dropout=0)
        config['positioning'].update(head_width=8,rssi_noise_db=0,device_shift_db=0,ap_dropout=0)
        raw=self.data(); ti,vi=group_split(raw); prep={"coordinates":{"center":[0,0],"shared_scale":2}}
        classes=[{"class_id":0,"building":0,"floor":1}]
        training=PaddedDataset(raw,ti,prep,classes,'train'); validation=PaddedDataset(raw,vi,prep,classes,'validation')
        fit_reference_points(config,training)
        coordinates={tuple(r['xy']) for r in config['positioning']['references']}
        self.assertFalse(coordinates & {tuple(v) for v in validation.xy.numpy()})
        with self.assertRaises(ValueError):fit_reference_points(config,validation)
        network=build_model(config,1).eval()
        hybrid=RefinedKANMixer(network,{'fit_scope':'internal_train_only','signal':db_features(training.features),'xy':training.xy,
            'class_ids':training.classes,'settings':{'neighbors':3,'distance_power':2,
            'neighbor_class_weight':0,'neural_coordinate_weight':.75}}).eval()
        with torch.inference_mode():result=hybrid(validation.features)
        self.assertTrue(torch.isfinite(result['xy']).all());self.assertEqual(tuple(result['xy'].shape),(8,2))

    def test_hcxy_mapping_retains_source_line_and_ap_order(self):
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            raw_root=Path(temporary);base=raw_root/'sodindoorloc/hcxy';base.mkdir(parents=True)
            (base/'mapxy_HCXY.txt').write_text('7 1.5 2.5\n9 3 4\n',encoding='utf-8')
            header='RP\t'+'\t'.join(f'MAC{i}' for i in range(1,348))+'\n'
            (base/'train_HCXY.txt').write_text(header+'9\t'+'\t'.join(['-70']*347)+'\n'+'7\t'+'\t'.join(['100']*347)+'\n',encoding='utf-8')
            data=read_supplied('hcxy','train',raw_root)
            np.testing.assert_array_equal(data.xy,[[3,4],[1.5,2.5]])
            self.assertEqual(data.source_rows.tolist(),[2,3]);self.assertEqual(data.ap_order[0],'MAC1')
            self.assertEqual(data.floors.tolist(),[4,4]);self.assertEqual(data.buildings.tolist(),[2,2])
            self.assertTrue((data.rssi[1]==100).all())

    def test_constant_classification_is_explicitly_descriptive(self):
        raw=self.data(); prep={'coordinates':{'center':[0,0],'shared_scale':1,'units':'metres_assumed'}}
        classes=[{'class_id':0,'building':0,'floor':1}]
        data=PaddedDataset(raw,np.arange(40),prep,classes,'test')
        result=measure(data,np.zeros(40,dtype=int),data.xy_original,SimpleNamespace(classes=classes,preprocessing=prep))
        self.assertEqual(result['mean_2d'],0);self.assertEqual(result['coordinate_units'],'metres_assumed')
        self.assertFalse(any(result['classification_applicability'].values()))

    def test_tuji_native_semantics_and_device_is_metadata_only(self):
        import csv
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            base=Path(temporary)/'tuji';base.mkdir()
            header=[f'WAP{i:03d}' for i in range(1,311)]+[
                'LONGITUDE','LATITUDE','FLOOR','BUILDINGID','SPACEID','RELATIVEPOSITION']
            with (base/'TUJI_training.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(header)
                writer.writerow([-101]+[100]*309+[1.5,2.5,0,4,8,5])
                writer.writerow([-60]*310+[1.5,2.5,0,4,8,1])
            raw=read_supplied('tuji','train',Path(temporary))
            self.assertEqual(raw.floors.tolist(),[4,4]);self.assertEqual(raw.buildings.tolist(),[8,8])
            self.assertEqual(raw.metadata['device_id'].tolist(),[5,1]);self.assertEqual(raw.rp.tolist(),[0,0])
            data=PaddedDataset(raw,np.arange(2),{'coordinates':{'center':[0,0],'shared_scale':1}},
                [{'class_id':0,'floor':4,'building':8}],'train')
            self.assertEqual(len(data[0]),3);self.assertEqual(data.source_rows.tolist(),[2,3])
            self.assertAlmostEqual(float(data.features[0,0,0]),3/104)
            self.assertEqual(float(data.features[0,0,1]),1)
            self.assertTrue((data.features[:,310:]==0).all())

    def test_syl_actual_label_mapping_and_no_ap_namespace_merge(self):
        import csv
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            base=Path(temporary)/'sodindoorloc/syl';base.mkdir(parents=True)
            header=[f'MAC{i}' for i in range(1,364)]+[
                'LONGITUDE','LATITUDE','FloorID','BuildingID','SceneID','UserID','PhoneID','SampleTimes']
            with (base/'SYL_Test.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(header)
                writer.writerow([-70]*363+[12,13,4,3,1,2,7,10])
            raw=read_supplied('syl','test',Path(temporary))
            self.assertEqual(raw.ap_order[-1],'MAC363');self.assertEqual(raw.buildings.tolist(),[3])
            self.assertEqual(raw.floors.tolist(),[4]);np.testing.assert_array_equal(raw.xy,[[12,13]])
            self.assertEqual(raw.metadata['device_id'].tolist(),[7]);self.assertEqual(raw.metadata['sample_times'].tolist(),[10])


if __name__=='__main__':unittest.main()
