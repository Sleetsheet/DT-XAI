"""Temporary-fixture software checks; not an Edge-IIoTset benchmark.
Run from the project directory: python tests/smoke_test.py
No physical USB enumeration is performed by these checks.
"""
from pathlib import Path
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def no_errors(app, stage):
    assert not app.exception, (stage, [e.value for e in app.exception])
    print('PASS', stage)


def main():
    with tempfile.TemporaryDirectory(prefix='verdant-check-') as tmp:
        os.environ['STATE_DIR'] = str(Path(tmp) / 'state')
        os.environ['DATA_PATH'] = str(Path(tmp) / 'fixture.csv')
        os.environ['APP_PASSWORD'] = 'test-only-private-password'
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
        no_errors(app, 'sign-in view')
        app.text_input(key='password_input').set_value(os.environ['APP_PASSWORD'])
        next(b for b in app.button if b.label == 'Enter workspace').click().run(timeout=60)
        no_errors(app, 'valid sign-in and missing-data workspace')
        assert any('ready for its data' in x.value for x in app.info)
        app.radio(key='navigation').set_value('Security monitor').run(timeout=60)
        no_errors(app, 'security independent of CSV')

        rng = np.random.default_rng(42)
        types = np.repeat(['Normal','DDoS_TCP','SQL_injection'], 100)
        data = pd.DataFrame({f'fixture_feature_{i}': rng.normal(size=300)+(types!='Normal')*(i%3) for i in range(15)})
        data['Attack_type'] = types
        data['Attack_label'] = (types!='Normal').astype(int)
        data['frame.time'] = 'discarded'
        data['tcp.srcport'] = 12345
        data['constant'] = 1
        data.to_csv(os.environ['DATA_PATH'], index=False)

        app.radio(key='navigation').set_value('Overview').run(timeout=60)
        no_errors(app, 'training, overview and automatic SHAP job')
        record = next(s for s in app.selectbox if s.label == 'Record ID')
        record.set_value(int(record.options[0][1:])).run(timeout=60)
        no_errors(app, 'local record explanation')
        next(s for s in app.selectbox if s.label == 'Status').set_value('Investigating').run(timeout=60)
        no_errors(app, 'persistent triage')

        for name in ['Threat explorer','Zero-day lab','Scenario studio','Data & methodology']:
            app.radio(key='navigation').set_value(name).run(timeout=60)
            no_errors(app, name)
            if name == 'Scenario studio':
                assert len(app.slider) == 12
                next(b for b in app.button if b.label == 'Normal example').click().run(timeout=60)
                no_errors(app, 'Normal example callback')
                app.slider[0].set_value(app.slider[0].min).run(timeout=60)
                no_errors(app, 'automatic scenario reassessment')
        app.selectbox(key='holdout').set_value('DDoS_TCP').run(timeout=60)
        no_errors(app, 'holdout switching')

        import security_monitor as sec
        sec.save_setting('usb_enabled', True)
        sec.save_setting('usb_baseline', None)
        with patch('security_monitor.inventory', return_value=['Keyboard']):
            sec.current_usb.clear()
            assert sec.usb_status()['new'] == []
        with patch('security_monitor.inventory', return_value=['Keyboard','Test phone']):
            sec.current_usb.clear()
            assert sec.usb_status()['new'] == ['Test phone']
        with patch('security_monitor.inventory', side_effect=RuntimeError('Simulated unavailable inventory')):
            sec.current_usb.clear()
            assert not sec.usb_status()['supported']
        sec.save_setting('usb_enabled', False)
        sec.current_usb.clear()
        assert sec.login_risk(5, 3, False, True)[0] == 100
        assert sec.risk_level(40) == 'Medium' and sec.risk_level(70) == 'Medium'
        print('PASS mocked USB baseline, change detection, unavailable state, score boundaries')

        next(b for b in app.button if b.label == 'Sign out').click().run(timeout=60)
        no_errors(app, 'sign-out')
        for _ in range(5):
            app.text_input(key='password_input').set_value('incorrect-password')
            next(b for b in app.button if b.label == 'Enter workspace').click().run(timeout=60)
            assert any('Incorrect password' in e.value for e in app.error)
        app.text_input(key='password_input').set_value(os.environ['APP_PASSWORD'])
        next(b for b in app.button if b.label == 'Enter workspace').click().run(timeout=60)
        assert any('Too many attempts' in e.value for e in app.error)
        no_errors(app, 'source-based lockout')
    print('All smoke checks passed. Temporary fixture removed; no real-data performance claim.')


if __name__ == '__main__':
    main()
