"""The native engine against S3, served by a local moto server."""

import json
import os

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from gammalake import DeltaOptions, GammaFeatureLake
from gammalake.native import NativeLake

moto_server = pytest.importorskip("moto.server")
moto_auth = pytest.importorskip("moto.core.authorization")
boto3 = pytest.importorskip("boto3")

BUCKET = "lake-bucket"
SCHEMA = pa.schema([("timestamp", pa.timestamp("us", "UTC")), ("symbol", pa.string())])
POLICY = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "*"}]}


@pytest.fixture(scope="module")
def server():
    """An endpoint with a bucket, and the key of an IAM user allowed to use it."""
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        setup = {"endpoint_url": endpoint, "region_name": "us-east-1", "aws_access_key_id": "setup", "aws_secret_access_key": "setup"}
        iam = boto3.client("iam", **setup)
        iam.create_user(UserName="lake")
        iam.put_user_policy(UserName="lake", PolicyName="s3", PolicyDocument=json.dumps(POLICY))
        key = iam.create_access_key(UserName="lake")["AccessKey"]
        boto3.client("s3", **setup).create_bucket(Bucket=BUCKET)
        yield endpoint, key["AccessKeyId"], key["SecretAccessKey"]
    finally:
        server.stop()


@pytest.fixture
def s3(server, tmp_path, monkeypatch):
    """The server, with no AWS settings in the environment."""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    return server


def options(endpoint, key_id=None, secret=None, **extra):
    keys = {} if key_id is None else {"AWS_ACCESS_KEY_ID": key_id, "AWS_SECRET_ACCESS_KEY": secret}
    return {"AWS_ENDPOINT_URL": endpoint, "AWS_ALLOW_HTTP": "true", "AWS_REGION": "us-east-1", **keys, **extra}


def frame():
    return pl.DataFrame(
        {
            "timestamp": [dt for dt in pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1d", eager=True)],
            "symbol": ["a", "a", "a"],
            "f": [1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_a_lake_on_s3_is_created_written_and_reopened(s3):
    endpoint, key_id, secret = s3
    storage_options = options(endpoint, key_id, secret)
    base_path = f"s3://{BUCKET}/lakes/one"
    delta_options = DeltaOptions(storage_options=storage_options)
    uninitialized = GammaFeatureLake(base_path=base_path, delta_options=delta_options)
    assert not uninitialized._is_initialized

    lake = uninitialized.initialize()
    lake.add_features(frame(), owner="me")
    expected = lake.read(["f"])

    reopened = GammaFeatureLake(base_path=base_path, delta_options=delta_options)
    assert reopened._is_initialized
    assert_frame_equal(reopened.read(["f"]), expected)
    assert expected["f"].to_list() == [1.0, 2.0, 3.0]
    assert secret not in repr(reopened)


def test_credentials_come_from_the_environment_when_options_omit_them(s3, tmp_path, monkeypatch):
    endpoint, key_id, secret = s3
    # With a custom endpoint, the AWS credential chain runs only when forced.
    storage_options = options(endpoint, AWS_FORCE_CREDENTIAL_LOAD="true")
    with pytest.raises(RuntimeError, match="credential"):
        NativeLake.create(f"s3://{BUCKET}/lakes/profile", SCHEMA, delta_options=DeltaOptions(storage_options=storage_options))
    (tmp_path / "credentials").write_text(f"[lake]\naws_access_key_id = {key_id}\naws_secret_access_key = {secret}\n")
    monkeypatch.setenv("AWS_PROFILE", "lake")
    lake = NativeLake.create(f"s3://{BUCKET}/lakes/profile", SCHEMA, delta_options=DeltaOptions(storage_options=storage_options))
    lake.add_features(frame())
    assert lake.read(["f"])["f"].to_list() == [1.0, 2.0, 3.0]


def test_a_wrong_key_is_refused_without_revealing_it(s3):
    endpoint, key_id, _ = s3
    bogus = "not-the-secret-0123456789"
    # moto checks signatures of only some requests correctly, so authentication is on just where a request must fail.
    with moto_auth.enable_iam_authentication(), pytest.raises(RuntimeError) as error:
        NativeLake.create(
            f"s3://{BUCKET}/lakes/bogus",
            SCHEMA,
            delta_options=DeltaOptions(storage_options=options(endpoint, key_id, bogus)),
        )
    assert bogus not in str(error.value)


def test_writers_holding_stale_handles_keep_each_others_commits(s3):
    endpoint, key_id, secret = s3
    storage_options = options(endpoint, key_id, secret)
    base_path = f"s3://{BUCKET}/lakes/shared"
    delta_options = DeltaOptions(storage_options=storage_options)
    first = NativeLake.create(base_path, SCHEMA, delta_options=delta_options)
    second = NativeLake(base_path, delta_options=delta_options)
    first.add_features(frame())
    second.add_features(frame().rename({"f": "g"}))
    result = NativeLake(base_path, delta_options=delta_options).read(["f", "g"])
    assert result["f"].to_list() == result["g"].to_list() == [1.0, 2.0, 3.0]
