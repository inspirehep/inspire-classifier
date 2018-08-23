from marshmallow import Schema, fields, INCLUDE


class ClassifierInputSerializer(Schema):
    class Meta:
        unknown = INCLUDE
    title = fields.Str(required=True)
    abstract = fields.Str(required=True)


class ClassifierOutputSerializer(Schema):
    score1 = fields.Float(attribute='score_a', required=True)
    score2 = fields.Float(attribute='score_b', required=True)
    score3 = fields.Float(attribute='score_c', required=True)
