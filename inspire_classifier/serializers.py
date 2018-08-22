from marshmallow import Schema, fields


class ClassifierInputSerializer(Schema):
    # TODO: set max size?
    title = fields.Str(required=True)
    abstract = fields.Str(required=True)


class ClassifierOutputSerializer(Schema):
    score1 = fields.Integer(attribute='score_a', required=True)
    score2 = fields.Integer(attribute='score_b', required=True)
    score3 = fields.Integer(attribute='score_c', required=True)
