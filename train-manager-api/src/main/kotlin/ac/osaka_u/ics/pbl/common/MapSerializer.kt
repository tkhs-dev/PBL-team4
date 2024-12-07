package ac.osaka_u.ics.pbl.common

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.*

object ParameterMapSerializer : KSerializer<Map<String, Any>> {
    override val descriptor: SerialDescriptor = MapSerializer(String.serializer(), JsonElement.serializer()).descriptor

    override fun serialize(encoder: Encoder, value: Map<String, Any>) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: throw SerializationException("This serializer can be used only with JSON")

        // MapをJsonObjectに変換
        val jsonMap = value.mapValues { (_, v) ->
            when (v) {
                is String -> JsonPrimitive(v)
                is Number -> JsonPrimitive(v)
                is Boolean -> JsonPrimitive(v)
                is List<*> -> JsonArray(v.map { item ->
                    when (item) {
                        is String -> JsonPrimitive(item)
                        is Number -> JsonPrimitive(item)
                        is Boolean -> JsonPrimitive(item)
                        else -> throw SerializationException("Unsupported list item type: ${item?.javaClass}")
                    }
                })
                else -> throw SerializationException("Unsupported value type: ${v.javaClass}")
            }
        }
        jsonEncoder.encodeJsonElement(JsonObject(jsonMap))
    }

    override fun deserialize(decoder: Decoder): Map<String, Any> {
        val jsonDecoder = decoder as? JsonDecoder
            ?: throw SerializationException("This serializer can be used only with JSON")

        val jsonObject = jsonDecoder.decodeJsonElement().jsonObject

        // JsonObjectをMapに変換
        return jsonObject.mapValues { (_, jsonElement) ->
            when (jsonElement) {
                is JsonPrimitive -> when {
                    jsonElement.isString -> jsonElement.content
                    jsonElement.booleanOrNull != null -> jsonElement.boolean
                    jsonElement.intOrNull != null -> jsonElement.int
                    jsonElement.doubleOrNull != null -> jsonElement.double
                    else -> throw SerializationException("Unsupported primitive type: $jsonElement")
                }
                is JsonArray -> jsonElement.map { item ->
                    when {
                        item is JsonPrimitive && item.isString -> item.content
                        item is JsonPrimitive && item.booleanOrNull != null -> item.boolean
                        item is JsonPrimitive && item.intOrNull != null -> item.int
                        item is JsonPrimitive && item.doubleOrNull != null -> item.double
                        else -> throw SerializationException("Unsupported array item type: $item")
                    }
                }
                else -> throw SerializationException("Unsupported value type: $jsonElement")
            }
        }
    }
}