"""E4: a fixed English template using every feature (plan §5). Template and name tables live in the config."""

import string

from src.encoders.common import column_values, is_missing, map_column, rounded_text

NEEDS_FIT = False


class _PlainFields(string.Formatter):
    """`{tcp.flags}` names the column "tcp.flags" (Edge-IIoTset names contain dots), not an attribute lookup."""

    def get_field(self, field_name, args, kwargs):
        return kwargs[field_name], field_name


_FORMATTER = _PlainFields()


def _name(value, table, unknown, decimals, na_token):
    """Readable name for a code; sub-protocol codes like 10.16 use their master code (10)."""
    if is_missing(value):
        return na_token
    return table.get(int(value), unknown.format(value=rounded_text(value, decimals, na_token)))


def encode(df, cfg, state):
    fmt, sent = cfg["format"], cfg["sentence"]
    decimals, na = fmt["float_decimals"], fmt["na_token"]
    features = cfg["data"]["columns"]["features"]
    values = column_values(df, features)
    fields = {c: map_column(values[c], lambda v: rounded_text(v, decimals, na)) for c in features}
    for name, spec in sent["derived"].items():
        table = cfg[spec["names"]]
        fields[name] = map_column(values[spec["source"]],
                                  lambda v, t=table, u=spec["unknown"]: _name(v, t, u, decimals, na))
    template = fmt["token_joiner"].join(sent["template"])
    keys = list(fields)
    return [_FORMATTER.vformat(template, (), dict(zip(keys, row))) for row in zip(*fields.values())]
