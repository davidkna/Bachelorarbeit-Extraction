import './index.scss'
import React, { useState, useEffect } from 'react'
import { render } from 'react-dom'
import {
  Card,
  Button,
  Navbar,
  Badge,
  Container,
  Form,
  Popover,
  OverlayTrigger,
  ListGroup,
  Spinner,
} from 'react-bootstrap'

// https://attacomsian.com/blog/javascript-base64-encode-decode
function btoaSafe(text: string): string {
  // first we use encodeURIComponent to get percent-encoded UTF-8,
  // then we convert the percent encodings into raw bytes which
  // can be fed into btoa.
  return btoa(
    encodeURIComponent(text).replace(/%([0-9A-F]{2})/g, function toSolidBytes(match, p1) {
      return String.fromCharCode(parseInt(p1, 16))
    })
  )
}

const THRESHOLDS = {
  word_embedding: 0.728,
  sent_embedding: 0.416,
  jaccard: 0.078,
  levenshtein: 0.259,
  relation: 1,
}

function CaseNebenentscheidungen({ refs, grund, method }) {
  let bodyText = 'Kein Textkörper'

  if (grund.self.length > 0) {
    bodyText = grund.self.map((i, idx) => <CaseRow key={idx} refs={refs} method={method} row={i} />)

    if (grund.subsections.length > 0) {
      bodyText = (
        <details>
          <summary>Textkörper</summary>
          {bodyText}
        </details>
      )
    }
  }

  return (
    <Card as="details">
      <Card.Header as="summary">Unterpunkt {grund.number}</Card.Header>
      <Card.Body>
        {bodyText}
        {grund.subsections.map((i, idx) => (
          <CaseNebenentscheidungen key={idx} refs={refs} grund={i} method={method} />
        ))}
      </Card.Body>
    </Card>
  )
}

function CaseRow({ refs, row, method }) {
  return (
    <>
      {row.line_no !== null ? `(${row.line_no})` : null}
      <ListGroup.Item>
        {row.sentences.map(sent => {
          return <Sent key={sent} text={sent} refs={refs} method={method} />
        })}
      </ListGroup.Item>
    </>
  )
}

function Sent({ refs, text, method }) {
  const maxMatches = Math.max(
    1,
    Math.max(
      ...Object.values(refs).map(i =>
        i[method] !== null && i[method] !== undefined ? i[method].length : 0
      )
    )
  )
  const matches =
    text in refs && refs[text][method] !== undefined && refs[text][method] !== null
      ? refs[text][method]
      : []
  const opacity = matches.length / maxMatches

  const outVal = (
    <ListGroup.Item
      id={btoaSafe(text)}
      key={btoaSafe(text)}
      style={{
        background: `rgba(25,135,84, ${opacity})`,
      }}
    >
      <div className="grid">
        <div className={`g-col-${matches.length > 0 ? "11" : "12"}`}>
          {method == 'jaccard' && matches.length > 0
            ? refs[text].wordwise.map((i, idx) => {
                const Tag = i.matches.length > 0 ? 'b' : 'span'
                return <Tag key={idx}>{i.word}</Tag>
              })
            : text}
        </div>
        {matches.length > 0 ? (
          <div className="g-col-1">
            <Badge bg="primary">{matches.length}</Badge>
          </div>
        ) : null}
      </div>
    </ListGroup.Item>
  )
  if (matches.length == 0) {
    return outVal
  }

  const ranking = matches.sort((a, b) => b.score - a.score)

  const popover = (
    <Popover id={`popover-${btoaSafe(text)}`}>
      <Popover.Header>Erkannte Referenzen</Popover.Header>
      {ranking.map(item => {
        return (
          <ListGroup.Item>
            <a href={`#${btoaSafe(item.text)}`}>
              {method === 'jaccard'
                ? refs[item.text].wordwise.map((i, idx) => {
                    const Tag = i.matches.includes(text) ? 'b' : 'span'
                    return <Tag key={idx}>{i.word}</Tag>
                  })
                : item.text}
            </a>
            <Badge bg="primary">{item.score.toFixed(3)}</Badge>
          </ListGroup.Item>
        )
      })}
    </Popover>
  )

  return (
    <OverlayTrigger trigger="click" placement="bottom" overlay={popover}>
      {outVal}
    </OverlayTrigger>
  )
}

function Section({ item, refs, method }) {
  if (item === null || item === undefined) {
    return null
  }
  return Object.entries(item).map(([k, v]) => {
    if (k === 'common' || k === 'slug' || k === 'ranking' || k === 'sentences' || k === 'line_no') {
      return null
    }

    if (k == 'nebenentscheidungen') {
      return (
        <Card as="details">
          <Card.Header as="summary">{k}</Card.Header>
          {v.map((grund, i) => (
            <CaseNebenentscheidungen key={i} refs={refs} grund={grund} method={method} />
          ))}
        </Card>
      )
    }

    if (Array.isArray(v)) {
      if (v.length === 0) {
        return (
          <Card as="details">
            <Card.Header as="summary">{k}</Card.Header>
          </Card>
        )
      }
      return (
        <Card as="details">
          <Card.Header as="summary">{k}</Card.Header>
          {v.map((row, i) => (
            <CaseRow key={i} refs={refs} row={row} method={method} />
          ))}
        </Card>
      )
    }
    if (v === null || v === undefined || 'subsections' in v) {
      return null
    } else {
      return (
        <Card as="details">
          <Card.Header as="summary">{k}</Card.Header>
          <Card.Body>
            <Section item={v} refs={refs} method={method} />
          </Card.Body>
        </Card>
      )
    }
  })
}

function Document({ docId, method }) {
  const [loading, setLoading] = useState(false)
  const [structure, loadStruct] = useState()
  const [references, loadRefs] = useState({})

  useEffect(() => {
    console.log(docId)
    if (docId === null || docId === undefined) {
      return
    }

    setLoading('Lade Dokument…')

    const abort_controller = new AbortController()
    const signal = abort_controller.signal

    fetch(`http://localhost:8000/fetch/${docId}`, {
      mode: 'cors',
      signal,
    })
      .then(r => r.json())
      .then(body => {
        setLoading('Strukturerkennung…')
        return fetch('http://localhost:8000/structure', {
          method: 'POST',
          mode: 'cors',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
          signal,
        })
      })
      .then(r => r.json())
      .then(newStruct => loadStruct(newStruct))
    return function cleanup() {
      abort_controller.abort()
    }
  }, [docId])

  useEffect(() => {
    if (structure === null || structure === undefined) {
      return
    }

    const abort_controller = new AbortController()
    const signal = abort_controller.signal

    loadRefs({})
    setLoading('Referenzerkennung…')

    fetch(`http://localhost:8000/semantic_references?${method}=${THRESHOLDS[method]}`, {
      method: 'POST',
      mode: 'cors',
      //cache: "force-cache",
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(structure),
      signal,
    })
      .then(r => r.json())
      .then(newRefs => {
        loadRefs(newRefs)
        setLoading(false)
      })
      .catch(console.error)

    return function cleanup() {
      abort_controller.abort()
    }
  }, [structure, method])

  return (
    <>
      {loading ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '50vw',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Spinner animation="border" />
            <div>{loading}</div>
          </div>
        </div>
      ) : null}
      <div
        style={{
          display: loading ? 'none' : 'block',
        }}
      >
        <Section refs={references} item={structure} method={method} />
      </div>
    </>
  )
}

function App() {
  const [value, setValue] = useState('330285')
  const [method, setMethod] = useState('jaccard')
  const [doc, setDoc] = useState()
  return (
    <>
      <Navbar bg="light" fixed="top">
        <Container>
          <Form
            onSubmit={e => {
              e.preventDefault()
              setDoc({
                method,
                id: value,
              })
            }}
          >
            <div className="grid">
              <div className="g-col-8">
                <Form.Control
                  placeholder="Urteil ID"
                  type="search"
                  value={value}
                  onChange={e => setValue(e.target.value)}
                />
              </div>
              <div className="g-col-3">
                <Form.Select
                  value={method}
                  aria-label="Reference Method"
                  onChange={e => setMethod(e.target.value)}
                >
                  <option value="jaccard">Jaccard-Index</option>
                  <option value="relation">Relationen</option>
                  <option value="word_embedding">Wortembedding</option>
                  <option value="sent_embedding">Satzembedding</option>
                  <option value="levenshtein">Levenshtein</option>
                </Form.Select>
              </div>
              <div className="g-col-1">
                <Button variant="primary" type="submit">
                  Absenden
                </Button>
              </div>
            </div>
          </Form>
        </Container>
      </Navbar>
      <Container>{doc ? <Document docId={doc.id} method={doc.method} /> : null}</Container>
    </>
  )
}

render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
)
